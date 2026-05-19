import argparse
import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIONAL_DATA_DIR = os.path.join(BASE_DIR, "Data", "National_Team_Data")
RAW_MATCHES_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_matches_raw.csv")
PROCESSED_MATCHES_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_matches_processed.csv")
MODEL_CACHE = os.path.join(NATIONAL_DATA_DIR, "national_team_model_cache.pkl")
API_REPORT_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_api_sources.json")

ESPN_SCOREBOARD_API = "https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/scoreboard"
FOOTBALL_DATA_API_BASE = "https://api.football-data.org/v4"
MIN_TRAINING_ROWS = 80
MODEL_RANDOM_STATE = 42
CATEGORICAL_FEATURE_COLUMNS = ["competition", "stage"]
RESULT_LABELS = {"H", "D", "A"}


def d(start, end):
    return (start, end)


ESPN_HISTORICAL_COMPETITIONS = {
    "FIFA/World Cup": {
        "espn_id": "fifa.world",
        "priority": 1,
        "ranges": [
            d("2002-05-31", "2002-06-30"),
            d("2006-06-09", "2006-07-09"),
            d("2010-06-11", "2010-07-11"),
            d("2014-06-12", "2014-07-13"),
            d("2018-06-14", "2018-07-15"),
            d("2022-11-20", "2022-12-18"),
        ],
    },
    "UEFA/European Championship": {
        "espn_id": "uefa.euro",
        "priority": 2,
        "ranges": [
            d("2004-06-12", "2004-07-04"),
            d("2008-06-07", "2008-06-29"),
            d("2012-06-08", "2012-07-01"),
            d("2016-06-10", "2016-07-10"),
            d("2021-06-11", "2021-07-11"),
            d("2024-06-14", "2024-07-14"),
        ],
    },
    "CONMEBOL/Copa America": {
        "espn_id": "conmebol.america",
        "priority": 3,
        "ranges": [
            d("2011-07-01", "2011-07-24"),
            d("2015-06-11", "2015-07-04"),
            d("2016-06-03", "2016-06-26"),
            d("2019-06-14", "2019-07-07"),
            d("2021-06-13", "2021-07-10"),
            d("2024-06-20", "2024-07-14"),
        ],
    },
    "CONCACAF/Gold Cup": {
        "espn_id": "concacaf.gold",
        "priority": 4,
        "ranges": [
            d("2011-06-05", "2011-06-25"),
            d("2013-07-07", "2013-07-28"),
            d("2015-07-07", "2015-07-26"),
            d("2017-07-07", "2017-07-26"),
            d("2019-06-15", "2019-07-07"),
            d("2021-07-10", "2021-08-01"),
            d("2023-06-24", "2023-07-16"),
        ],
    },
    "CAF/Africa Cup of Nations": {
        "espn_id": "caf.nations",
        "priority": 5,
        "ranges": [
            d("2012-01-21", "2012-02-12"),
            d("2013-01-19", "2013-02-10"),
            d("2015-01-17", "2015-02-08"),
            d("2017-01-14", "2017-02-05"),
            d("2019-06-21", "2019-07-19"),
            d("2022-01-09", "2022-02-06"),
            d("2024-01-13", "2024-02-11"),
        ],
    },
    "UEFA/Nations League": {
        "espn_id": "uefa.nations",
        "priority": 6,
        "ranges": [
            d("2019-06-05", "2019-06-09"),
            d("2021-10-06", "2021-10-10"),
            d("2023-06-14", "2023-06-18"),
            d("2025-06-04", "2025-06-08"),
        ],
    },
}

ESPN_QUALIFIER_COMPETITIONS = {
    "FIFA/World Cup Qualifying - UEFA": {
        "espn_id": "fifa.worldq.uefa",
        "priority": 10,
        "ranges": [d("2024-09-01", "2026-03-31")],
    },
    "FIFA/World Cup Qualifying - CONMEBOL": {
        "espn_id": "fifa.worldq.conmebol",
        "priority": 11,
        "ranges": [d("2023-09-01", "2026-03-31")],
    },
    "FIFA/World Cup Qualifying - CONCACAF": {
        "espn_id": "fifa.worldq.concacaf",
        "priority": 12,
        "ranges": [d("2024-03-01", "2026-03-31")],
    },
}

UPCOMING_ESPN_COMPETITIONS = {
    **ESPN_HISTORICAL_COMPETITIONS,
    **ESPN_QUALIFIER_COMPETITIONS,
    "FIFA/Friendly": {"espn_id": "fifa.friendly", "priority": 20, "ranges": []},
}

FOOTBALL_DATA_COMPETITIONS = {
    "WC": "FIFA/World Cup",
    "EC": "UEFA/European Championship",
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Fetch national-team history, process rolling features, and train a predictor."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Train from the existing raw national-team CSV instead of calling APIs.",
    )
    parser.add_argument(
        "--world-cup-only",
        action="store_true",
        help="Only fetch/process World Cup matches. By default other major national competitions are included too.",
    )
    parser.add_argument(
        "--include-qualifiers",
        action="store_true",
        help="Also scan recent World Cup qualifying competitions from ESPN. This performs many more API calls.",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=os.getenv("FOOTBALL_DATA_API_TOKEN", "").strip(),
        help="Optional football-data.org token for supplemental national competition history.",
    )
    return parser.parse_args()


def normalize_team_key(name):
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stop_words = {"fc", "cf", "team", "national", "football", "soccer", "the"}
    parts = [part for part in text.split() if part and part not in stop_words]
    aliases = {
        "usa": "unitedstates",
        "us": "unitedstates",
        "u s": "unitedstates",
        "czechrepublic": "czechia",
        "turkiye": "turkey",
        "ivorycoast": "cotedivoire",
        "cotedivoire": "cotedivoire",
        "bosniaherzegovina": "bosniaandherzegovina",
        "curacao": "curacao",
    }
    key = "".join(parts)
    return aliases.get(key, key)


def make_prediction_key(match_date, competition, home_team, away_team):
    parsed = pd.to_datetime(match_date, utc=True, errors="coerce")
    if pd.isna(parsed):
        date_part = str(match_date)[:10]
    else:
        date_part = parsed.strftime("%Y-%m-%d")
    pair = sorted([normalize_team_key(home_team), normalize_team_key(away_team)])
    return f"{date_part}|{competition}|{pair[0]}|{pair[1]}"


def date_range(start_date, end_date):
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_json(url, headers=None, timeout=30):
    request = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_number(value, default=0.0):
    if value is None:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return default
    try:
        return float(match.group(0))
    except ValueError:
        return default


def competitor_stat(competitor, names):
    wanted = {name.lower() for name in names}
    for stat in competitor.get("statistics") or []:
        stat_name = str(stat.get("name", "")).lower()
        abbreviation = str(stat.get("abbreviation", "")).lower()
        if stat_name in wanted or abbreviation in wanted:
            return parse_number(stat.get("displayValue"))
    return 0.0


def _competitors_by_side(competitors):
    by_side = {}
    ordered = list(competitors or [])
    for competitor in ordered:
        side = str(competitor.get("homeAway", "")).strip().lower()
        if side in {"home", "away"}:
            by_side[side] = competitor
    if "home" not in by_side and ordered:
        by_side["home"] = ordered[0]
    if "away" not in by_side and len(ordered) > 1:
        by_side["away"] = ordered[1]
    return by_side


def parse_espn_event(event, competition_name, require_completed=True):
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    competition = competitions[0] or {}
    status_type = ((competition.get("status") or {}).get("type") or {})
    state = str(status_type.get("state", "")).strip().lower()
    completed = bool(status_type.get("completed")) or state == "post"
    if require_completed and not completed:
        return None

    by_side = _competitors_by_side(competition.get("competitors") or [])
    home = by_side.get("home")
    away = by_side.get("away")
    if not home or not away:
        return None

    home_team = ((home.get("team") or {}).get("displayName") or "").strip()
    away_team = ((away.get("team") or {}).get("displayName") or "").strip()
    if not home_team or not away_team:
        return None

    event_dt = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
    if pd.isna(event_dt):
        return None

    home_goals = parse_number(home.get("score"), default=None)
    away_goals = parse_number(away.get("score"), default=None)
    result = ""
    if completed and home_goals is not None and away_goals is not None:
        if home_goals > away_goals:
            result = "H"
        elif away_goals > home_goals:
            result = "A"
        elif bool(home.get("winner")) and not bool(away.get("winner")):
            result = "H"
        elif bool(away.get("winner")) and not bool(home.get("winner")):
            result = "A"
        else:
            result = "D"

    stage = str((event.get("season") or {}).get("slug", "") or "").strip()
    status_name = str(status_type.get("name", "") or status_type.get("description", "")).strip()
    venue = competition.get("venue") or {}
    is_neutral = competition.get("neutralSite")
    if is_neutral is None:
        is_neutral = is_likely_neutral_competition(competition_name)

    return {
        "match_id": str(event.get("id", "")).strip(),
        "match_datetime_utc": event_dt.isoformat(),
        "match_date": event_dt.strftime("%Y-%m-%d"),
        "competition": competition_name,
        "stage": stage or "unknown",
        "home_team": home_team,
        "away_team": away_team,
        "FTHG": int(home_goals) if home_goals is not None else None,
        "FTAG": int(away_goals) if away_goals is not None else None,
        "FTR": result,
        "home_shots": competitor_stat(home, {"totalShots", "SH"}),
        "away_shots": competitor_stat(away, {"totalShots", "SH"}),
        "home_sot": competitor_stat(home, {"shotsOnTarget", "ST"}),
        "away_sot": competitor_stat(away, {"shotsOnTarget", "ST"}),
        "home_penalties": home.get("shootoutScore"),
        "away_penalties": away.get("shootoutScore"),
        "status": status_name,
        "is_neutral_site": bool(is_neutral),
        "venue": str(venue.get("fullName", "") or venue.get("name", "") or "").strip(),
        "source": "espn",
    }


def is_likely_neutral_competition(competition_name):
    lowered = str(competition_name).lower()
    neutral_terms = ["world cup", "european championship", "copa america", "gold cup", "africa cup", "nations league"]
    return any(term in lowered for term in neutral_terms)


def selected_espn_competitions(world_cup_only=False, include_qualifiers=False):
    if world_cup_only:
        return {"FIFA/World Cup": ESPN_HISTORICAL_COMPETITIONS["FIFA/World Cup"]}
    competitions = dict(ESPN_HISTORICAL_COMPETITIONS)
    if include_qualifiers:
        competitions.update(ESPN_QUALIFIER_COMPETITIONS)
    return competitions


def fetch_espn_historical_matches(competitions):
    rows = []
    seen_event_ids = set()
    for competition_name, config in sorted(competitions.items(), key=lambda item: item[1]["priority"]):
        espn_id = config["espn_id"]
        for start_date, end_date in config.get("ranges", []):
            print(f"Fetching ESPN {competition_name}: {start_date} to {end_date}")
            for day in date_range(start_date, end_date):
                url = ESPN_SCOREBOARD_API.format(espn_id=espn_id) + f"?dates={day.strftime('%Y%m%d')}"
                try:
                    payload = fetch_json(url, timeout=30)
                except Exception as exc:
                    print(f"  ESPN fetch failed for {competition_name} {day}: {exc}")
                    continue
                for event in payload.get("events") or []:
                    event_id = str(event.get("id", "")).strip()
                    if event_id and event_id in seen_event_ids:
                        continue
                    parsed = parse_espn_event(event, competition_name, require_completed=True)
                    if not parsed:
                        continue
                    if event_id:
                        seen_event_ids.add(event_id)
                    rows.append(parsed)
    return rows


def fetch_football_data_historical_matches(api_token, competitions):
    if not api_token:
        return []

    rows = []
    headers = {"X-Auth-Token": api_token}
    target_names = set(competitions.keys())
    for competition_code, competition_name in FOOTBALL_DATA_COMPETITIONS.items():
        if competition_name not in target_names:
            continue
        for start_date, end_date in competitions[competition_name].get("ranges", []):
            query = urllib.parse.urlencode(
                {"dateFrom": start_date, "dateTo": end_date, "status": "FINISHED"}
            )
            url = f"{FOOTBALL_DATA_API_BASE}/competitions/{competition_code}/matches?{query}"
            try:
                payload = fetch_json(url, headers=headers, timeout=45)
            except urllib.error.HTTPError as error:
                if error.code == 401:
                    raise RuntimeError("football-data.org API token is invalid.") from error
                if error.code in {403, 404, 429}:
                    print(f"football-data.org unavailable for {competition_name} ({error.code}); using ESPN only.")
                    break
                continue
            except Exception as exc:
                print(f"football-data.org fetch failed for {competition_name}: {exc}")
                continue

            for match in payload.get("matches") or []:
                parsed = parse_football_data_match(match, competition_name, completed_only=True)
                if parsed:
                    rows.append(parsed)
    return rows


def parse_football_data_match(match, competition_name, completed_only=True):
    status = str(match.get("status", "")).strip().upper()
    if completed_only and status not in {"FINISHED", "FULL_TIME"}:
        return None
    home_team = ((match.get("homeTeam") or {}).get("name") or "").strip()
    away_team = ((match.get("awayTeam") or {}).get("name") or "").strip()
    parsed_dt = pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce")
    if not home_team or not away_team or pd.isna(parsed_dt):
        return None

    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}
    home_goals = full_time.get("home")
    away_goals = full_time.get("away")
    winner = str(score.get("winner", "")).strip().upper()
    result = ""
    if home_goals is not None and away_goals is not None:
        if home_goals > away_goals or winner == "HOME_TEAM":
            result = "H"
        elif away_goals > home_goals or winner == "AWAY_TEAM":
            result = "A"
        else:
            result = "D"

    return {
        "match_id": str(match.get("id", "")).strip(),
        "match_datetime_utc": parsed_dt.isoformat(),
        "match_date": parsed_dt.strftime("%Y-%m-%d"),
        "competition": competition_name,
        "stage": str(match.get("stage", "") or "unknown").strip().lower(),
        "home_team": home_team,
        "away_team": away_team,
        "FTHG": int(home_goals) if home_goals is not None else None,
        "FTAG": int(away_goals) if away_goals is not None else None,
        "FTR": result,
        "home_shots": 0.0,
        "away_shots": 0.0,
        "home_sot": 0.0,
        "away_sot": 0.0,
        "home_penalties": None,
        "away_penalties": None,
        "status": status,
        "is_neutral_site": is_likely_neutral_competition(competition_name),
        "venue": str(match.get("venue", "") or "").strip(),
        "source": "football-data.org",
    }


def dedupe_matches(rows):
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    source_order = {"espn": 0, "football-data.org": 1}
    frame["source_order"] = frame["source"].map(source_order).fillna(99)
    frame["dedupe_key"] = frame.apply(
        lambda row: make_prediction_key(
            row["match_date"], row["competition"], row["home_team"], row["away_team"]
        ),
        axis=1,
    )
    frame = frame.sort_values(["source_order", "match_datetime_utc", "dedupe_key"])
    frame = frame.drop_duplicates(subset=["dedupe_key"], keep="first")
    frame = frame.drop(columns=["source_order", "dedupe_key"])
    frame = frame.sort_values(["match_datetime_utc", "competition", "home_team"]).reset_index(drop=True)
    return frame


def default_team_state():
    return {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "home_games": 0,
        "away_games": 0,
        "home_goals_for": 0,
        "home_goals_against": 0,
        "away_goals_for": 0,
        "away_goals_against": 0,
        "recent_results": [],
        "recent_goals_for": [],
        "recent_goals_against": [],
        "elo": 1500.0,
    }


def default_h2h_state():
    return {"games": 0, "wins": 0, "draws": 0, "goals_for": 0, "goals_against": 0}


def clone_state(state):
    cloned = dict(state)
    for key in ["recent_results", "recent_goals_for", "recent_goals_against"]:
        cloned[key] = list(cloned.get(key, []))
    return cloned


def safe_div(numerator, denominator, default=0.0):
    try:
        denominator = float(denominator)
        if denominator == 0:
            return default
        return float(numerator) / denominator
    except Exception:
        return default


def recent_average(values, default=0.0):
    values = list(values or [])[-10:]
    if not values:
        return default
    return sum(float(value) for value in values) / len(values)


def recent_points(results):
    total = 0.0
    for result in list(results or [])[-10:]:
        if result == "W":
            total += 3.0
        elif result == "D":
            total += 1.0
    return total


def h2h_key(team_a, team_b):
    return f"{normalize_team_key(team_a)}||{normalize_team_key(team_b)}"


def stage_is_knockout(stage):
    text = str(stage or "").lower()
    if "group" in text or "qualifying" in text:
        return False
    return any(term in text for term in ["final", "semi", "quarter", "round", "knockout", "third"])


def team_feature_values(prefix, stats):
    games = float(stats.get("games", 0) or 0)
    wins = float(stats.get("wins", 0) or 0)
    draws = float(stats.get("draws", 0) or 0)
    losses = float(stats.get("losses", 0) or 0)
    points = float(stats.get("points", 0) or 0)
    gf = float(stats.get("goals_for", 0) or 0)
    ga = float(stats.get("goals_against", 0) or 0)
    home_games = float(stats.get("home_games", 0) or 0)
    away_games = float(stats.get("away_games", 0) or 0)
    return {
        f"{prefix}_games": games,
        f"{prefix}_win_rate": safe_div(wins, games),
        f"{prefix}_draw_rate": safe_div(draws, games),
        f"{prefix}_loss_rate": safe_div(losses, games),
        f"{prefix}_points_per_game": safe_div(points, games),
        f"{prefix}_goals_for_per_game": safe_div(gf, games),
        f"{prefix}_goals_against_per_game": safe_div(ga, games),
        f"{prefix}_goal_diff_per_game": safe_div(gf - ga, games),
        f"{prefix}_home_goals_for_per_game": safe_div(stats.get("home_goals_for", 0), home_games),
        f"{prefix}_home_goals_against_per_game": safe_div(stats.get("home_goals_against", 0), home_games),
        f"{prefix}_away_goals_for_per_game": safe_div(stats.get("away_goals_for", 0), away_games),
        f"{prefix}_away_goals_against_per_game": safe_div(stats.get("away_goals_against", 0), away_games),
        f"{prefix}_recent_points_per_game": recent_points(stats.get("recent_results", [])) / 10.0,
        f"{prefix}_recent_goals_for": recent_average(stats.get("recent_goals_for", [])),
        f"{prefix}_recent_goals_against": recent_average(stats.get("recent_goals_against", [])),
        f"{prefix}_elo": float(stats.get("elo", 1500.0) or 1500.0),
    }


def build_feature_row(home_team, away_team, competition, stage, is_neutral_site, team_state, h2h_state):
    home_stats = clone_state(team_state.get(home_team, default_team_state()))
    away_stats = clone_state(team_state.get(away_team, default_team_state()))
    h2h = h2h_state.get(h2h_key(home_team, away_team), default_h2h_state())

    row = {}
    row.update(team_feature_values("home", home_stats))
    row.update(team_feature_values("away", away_stats))
    row.update(
        {
            "games_diff": row["home_games"] - row["away_games"],
            "points_per_game_diff": row["home_points_per_game"] - row["away_points_per_game"],
            "goal_diff_per_game_diff": row["home_goal_diff_per_game"] - row["away_goal_diff_per_game"],
            "recent_points_per_game_diff": row["home_recent_points_per_game"] - row["away_recent_points_per_game"],
            "elo_diff": row["home_elo"] - row["away_elo"],
            "h2h_games": float(h2h.get("games", 0) or 0),
            "h2h_home_win_rate": safe_div(h2h.get("wins", 0), h2h.get("games", 0)),
            "h2h_draw_rate": safe_div(h2h.get("draws", 0), h2h.get("games", 0)),
            "h2h_goal_diff_per_game": safe_div(
                float(h2h.get("goals_for", 0) or 0) - float(h2h.get("goals_against", 0) or 0),
                h2h.get("games", 0),
            ),
            "is_world_cup": 1.0 if "world cup" in str(competition).lower() else 0.0,
            "is_knockout": 1.0 if stage_is_knockout(stage) else 0.0,
            "is_neutral_site": 1.0 if is_neutral_site else 0.0,
            "competition": competition,
            "stage": str(stage or "unknown").strip().lower() or "unknown",
        }
    )
    return row


def append_recent(stats, result, goals_for, goals_against):
    stats["recent_results"] = (list(stats.get("recent_results", [])) + [result])[-10:]
    stats["recent_goals_for"] = (list(stats.get("recent_goals_for", [])) + [float(goals_for)])[-10:]
    stats["recent_goals_against"] = (list(stats.get("recent_goals_against", [])) + [float(goals_against)])[-10:]


def update_elo(home_stats, away_stats, result, home_goals, away_goals, competition):
    home_elo = float(home_stats.get("elo", 1500.0) or 1500.0)
    away_elo = float(away_stats.get("elo", 1500.0) or 1500.0)
    expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
    actual_home = 1.0 if result == "H" else 0.0 if result == "A" else 0.5
    goal_margin = max(1.0, abs(float(home_goals) - float(away_goals)))
    importance = 1.45 if "world cup" in str(competition).lower() else 1.20
    k_factor = 22.0 * importance * (1.0 + min(goal_margin, 4.0) * 0.12)
    change = k_factor * (actual_home - expected_home)
    home_stats["elo"] = round(home_elo + change, 3)
    away_stats["elo"] = round(away_elo - change, 3)


def update_team_and_h2h_state(match, team_state, h2h_state):
    home = match["home_team"]
    away = match["away_team"]
    result = match["FTR"]
    home_goals = int(match["FTHG"])
    away_goals = int(match["FTAG"])
    competition = match["competition"]

    home_stats = team_state.setdefault(home, default_team_state())
    away_stats = team_state.setdefault(away, default_team_state())

    update_elo(home_stats, away_stats, result, home_goals, away_goals, competition)

    for stats, gf, ga, side in [(home_stats, home_goals, away_goals, "home"), (away_stats, away_goals, home_goals, "away")]:
        stats["games"] += 1
        stats["goals_for"] += gf
        stats["goals_against"] += ga
        stats[f"{side}_games"] += 1
        stats[f"{side}_goals_for"] += gf
        stats[f"{side}_goals_against"] += ga

    if result == "H":
        home_stats["wins"] += 1
        home_stats["points"] += 3
        away_stats["losses"] += 1
        append_recent(home_stats, "W", home_goals, away_goals)
        append_recent(away_stats, "L", away_goals, home_goals)
    elif result == "A":
        away_stats["wins"] += 1
        away_stats["points"] += 3
        home_stats["losses"] += 1
        append_recent(home_stats, "L", home_goals, away_goals)
        append_recent(away_stats, "W", away_goals, home_goals)
    else:
        home_stats["draws"] += 1
        away_stats["draws"] += 1
        home_stats["points"] += 1
        away_stats["points"] += 1
        append_recent(home_stats, "D", home_goals, away_goals)
        append_recent(away_stats, "D", away_goals, home_goals)

    home_h2h = h2h_state.setdefault(h2h_key(home, away), default_h2h_state())
    away_h2h = h2h_state.setdefault(h2h_key(away, home), default_h2h_state())
    home_h2h["games"] += 1
    home_h2h["goals_for"] += home_goals
    home_h2h["goals_against"] += away_goals
    away_h2h["games"] += 1
    away_h2h["goals_for"] += away_goals
    away_h2h["goals_against"] += home_goals
    if result == "H":
        home_h2h["wins"] += 1
    elif result == "A":
        away_h2h["wins"] += 1
    else:
        home_h2h["draws"] += 1
        away_h2h["draws"] += 1


def process_matches(raw_matches):
    required = {"match_datetime_utc", "competition", "home_team", "away_team", "FTHG", "FTAG", "FTR"}
    if not required.issubset(raw_matches.columns):
        missing = sorted(required - set(raw_matches.columns))
        raise ValueError("Raw national data is missing required columns: " + ", ".join(missing))

    frame = raw_matches.copy()
    frame["match_datetime_utc"] = pd.to_datetime(frame["match_datetime_utc"], utc=True, errors="coerce")
    frame = frame[frame["match_datetime_utc"].notna()]
    frame = frame[frame["FTR"].astype(str).str.strip().isin(RESULT_LABELS)]
    frame = frame[frame["FTHG"].notna() & frame["FTAG"].notna()]
    frame = frame.sort_values(["match_datetime_utc", "competition", "home_team"]).reset_index(drop=True)

    team_state = {}
    h2h_state = {}
    processed_rows = []
    for _, match in frame.iterrows():
        feature_row = build_feature_row(
            match["home_team"],
            match["away_team"],
            match["competition"],
            match.get("stage", "unknown"),
            bool(match.get("is_neutral_site", False)),
            team_state,
            h2h_state,
        )
        processed_rows.append(
            {
                **feature_row,
                "match_id": match.get("match_id", ""),
                "match_datetime_utc": match["match_datetime_utc"].isoformat(),
                "match_date": match["match_datetime_utc"].strftime("%Y-%m-%d"),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "FTHG": int(match["FTHG"]),
                "FTAG": int(match["FTAG"]),
                "FTR": str(match["FTR"]),
                "home_shots": float(match.get("home_shots", 0.0) or 0.0),
                "away_shots": float(match.get("away_shots", 0.0) or 0.0),
                "home_sot": float(match.get("home_sot", 0.0) or 0.0),
                "away_sot": float(match.get("away_sot", 0.0) or 0.0),
                "source": match.get("source", ""),
            }
        )
        update_team_and_h2h_state(match, team_state, h2h_state)

    processed = pd.DataFrame(processed_rows)
    snapshot = {
        "team_state": team_state,
        "h2h_state": h2h_state,
        "known_teams": sorted(team_state.keys()),
        "latest_match_datetime_utc": frame["match_datetime_utc"].max().isoformat() if not frame.empty else "",
    }
    return processed, snapshot


def data_fingerprint(path):
    digest = hashlib.sha256()
    if not os.path.exists(path):
        return ""
    digest.update(os.path.basename(path).encode("utf-8"))
    digest.update(str(os.path.getmtime(path)).encode("utf-8"))
    digest.update(str(os.path.getsize(path)).encode("utf-8"))
    return digest.hexdigest()


def train_model(processed, snapshot):
    if len(processed) < MIN_TRAINING_ROWS:
        raise ValueError(f"Need at least {MIN_TRAINING_ROWS} national matches to train; found {len(processed)}.")

    metadata_columns = {
        "match_id",
        "match_datetime_utc",
        "match_date",
        "home_team",
        "away_team",
        "FTHG",
        "FTAG",
        "FTR",
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "source",
    }
    feature_frame = processed[[col for col in processed.columns if col not in metadata_columns]].copy()
    feature_frame = pd.get_dummies(feature_frame, columns=CATEGORICAL_FEATURE_COLUMNS, dtype=float)
    train_columns = list(feature_frame.columns)

    label_encoder = LabelEncoder()
    y_result = label_encoder.fit_transform(processed["FTR"].astype(str))
    y_home_goals = processed["FTHG"].astype(float).reset_index(drop=True)
    y_away_goals = processed["FTAG"].astype(float).reset_index(drop=True)

    clf = RandomForestClassifier(
        n_estimators=360,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    home_goal_reg = RandomForestRegressor(
        n_estimators=260,
        min_samples_leaf=2,
        random_state=MODEL_RANDOM_STATE + 1,
        n_jobs=-1,
    )
    away_goal_reg = RandomForestRegressor(
        n_estimators=260,
        min_samples_leaf=2,
        random_state=MODEL_RANDOM_STATE + 2,
        n_jobs=-1,
    )

    clf.fit(feature_frame, y_result)
    home_goal_reg.fit(feature_frame, y_home_goals)
    away_goal_reg.fit(feature_frame, y_away_goals)

    holdout_accuracy = None
    if len(feature_frame) >= 140:
        split_idx = int(len(feature_frame) * 0.80)
        holdout_clf = RandomForestClassifier(
            n_estimators=260,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1,
        )
        holdout_clf.fit(feature_frame.iloc[:split_idx], y_result[:split_idx])
        holdout_accuracy = float(holdout_clf.score(feature_frame.iloc[split_idx:], y_result[split_idx:]))

    return {
        "model_version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "fingerprint": data_fingerprint(PROCESSED_MATCHES_FILE),
        "clf": clf,
        "result_label_encoder": label_encoder,
        "home_goal_reg": home_goal_reg,
        "away_goal_reg": away_goal_reg,
        "train_columns": train_columns,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "snapshot": snapshot,
        "training_rows": int(len(processed)),
        "holdout_accuracy": holdout_accuracy,
    }


def resolve_team_name(raw_name, known_teams):
    raw = str(raw_name or "").strip()
    if raw in known_teams:
        return raw
    key = normalize_team_key(raw)
    by_key = {normalize_team_key(team): team for team in known_teams}
    if key in by_key:
        return by_key[key]
    contains = [team for team in known_teams if key and key in normalize_team_key(team)]
    if len(contains) == 1:
        return contains[0]
    reverse_contains = [team for team in known_teams if normalize_team_key(team) and normalize_team_key(team) in key]
    if len(reverse_contains) == 1:
        return reverse_contains[0]
    return raw


def build_prediction_feature_frame(home_team, away_team, competition, stage, is_neutral_site, snapshot):
    known_teams = snapshot.get("known_teams", [])
    home = resolve_team_name(home_team, known_teams)
    away = resolve_team_name(away_team, known_teams)
    row = build_feature_row(
        home,
        away,
        competition,
        stage,
        is_neutral_site,
        snapshot.get("team_state", {}),
        snapshot.get("h2h_state", {}),
    )
    return pd.DataFrame([row]), home, away


def align_feature_frame(raw_feature_frame, bundle):
    categorical_columns = bundle.get("categorical_feature_columns", CATEGORICAL_FEATURE_COLUMNS)
    present_categorical = [col for col in categorical_columns if col in raw_feature_frame.columns]
    frame = pd.get_dummies(raw_feature_frame, columns=present_categorical, dtype=float)
    return frame.reindex(columns=bundle["train_columns"], fill_value=0.0)


def load_model_bundle(path=MODEL_CACHE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"National model cache not found at {path}. Run Process_National_Team_Data.py first."
        )
    return joblib.load(path)


def probability_jitter(probabilities, key, max_delta):
    h = max(0.0, float(probabilities.get("H", 0.0)))
    d = max(0.0, float(probabilities.get("D", 0.0)))
    a = max(0.0, float(probabilities.get("A", 0.0)))
    total = h + d + a
    if total <= 0:
        return {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}
    h, d, a = h / total, d / total, a / total
    seed = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    delta = rng.uniform(-max_delta, max_delta)
    h = max(0.0, min(1.0, h + delta))
    a = max(0.0, min(1.0, a - delta))
    total = h + d + a
    return {"H": h / total, "D": d / total, "A": a / total}


def write_api_report(competitions, used_football_data):
    report = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "summary": [
            "ESPN scoreboard API is usable for national-team historical data and upcoming fixtures.",
            "The existing ESPN soccer endpoint returned historical FIFA World Cup data back to 2002 and scheduled 2026 World Cup fixtures.",
            "football-data.org is wired as an optional supplemental source for national competitions when FOOTBALL_DATA_API_TOKEN is available.",
            "TheSportsDB eventsnextleague endpoint used by cup predictions is suitable as a narrow upcoming-fixture fallback, but it is not used here for broad historical national-team training.",
        ],
        "espn_competitions": {
            name: {"espn_id": config["espn_id"], "ranges": config.get("ranges", [])}
            for name, config in competitions.items()
        },
        "football_data_enabled": bool(used_football_data),
        "outputs": {
            "raw_matches": RAW_MATCHES_FILE,
            "processed_matches": PROCESSED_MATCHES_FILE,
            "model_cache": MODEL_CACHE,
        },
    }
    os.makedirs(NATIONAL_DATA_DIR, exist_ok=True)
    with open(API_REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def run_pipeline(args):
    os.makedirs(NATIONAL_DATA_DIR, exist_ok=True)
    competitions = selected_espn_competitions(
        world_cup_only=args.world_cup_only,
        include_qualifiers=args.include_qualifiers,
    )

    if args.skip_fetch:
        if not os.path.exists(RAW_MATCHES_FILE):
            raise FileNotFoundError(f"Cannot --skip-fetch because {RAW_MATCHES_FILE} does not exist.")
        raw_matches = pd.read_csv(RAW_MATCHES_FILE)
    else:
        rows = fetch_espn_historical_matches(competitions)
        fd_rows = fetch_football_data_historical_matches(args.api_token, competitions)
        rows.extend(fd_rows)
        raw_matches = dedupe_matches(rows)
        raw_matches.to_csv(RAW_MATCHES_FILE, index=False)
        write_api_report(competitions, used_football_data=bool(fd_rows))

    if raw_matches.empty:
        raise ValueError("No national-team historical matches were fetched or loaded.")

    processed, snapshot = process_matches(raw_matches)
    processed.to_csv(PROCESSED_MATCHES_FILE, index=False)
    bundle = train_model(processed, snapshot)
    joblib.dump(bundle, MODEL_CACHE)

    print("\nNational-team predictor generated")
    print(f"Historical matches loaded: {len(raw_matches)}")
    print(f"Processed training rows: {len(processed)}")
    print(f"Known national teams: {len(snapshot.get('known_teams', []))}")
    print(f"Raw data: {RAW_MATCHES_FILE}")
    print(f"Processed data: {PROCESSED_MATCHES_FILE}")
    print(f"Model cache: {MODEL_CACHE}")
    if bundle.get("holdout_accuracy") is not None:
        print(f"Chronological holdout accuracy: {bundle['holdout_accuracy']:.3f}")
    return bundle


def main():
    args = parse_cli_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
