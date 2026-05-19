import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import urllib.parse
import urllib.request
import unicodedata
from datetime import UTC, datetime, timedelta

import joblib
import pandas as pd


if __name__ == "__main__":
    sys.modules.setdefault("Process_National_Team_Data", sys.modules[__name__])


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIONAL_DATA_DIR = os.path.join(BASE_DIR, "Data", "National_Team_Data")
RAW_MATCHES_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_recent_matches_raw.csv")
PROCESSED_MATCHES_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_recent_context.csv")
MODEL_CACHE = os.path.join(NATIONAL_DATA_DIR, "national_team_model_cache.pkl")
API_REPORT_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_api_sources.json")
FIFA_RANKINGS_FILE = os.path.join(NATIONAL_DATA_DIR, "fifa_rankings.json")
SQUAD_VALUES_FILE = os.path.join(NATIONAL_DATA_DIR, "national_team_squad_values.json")

ESPN_SCOREBOARD_API = "https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/scoreboard"
FOOTBALL_DATA_API_BASE = "https://api.football-data.org/v4"
FOOTBALLDATA_IO_BASE = "https://footballdata.io/api/v1"
SPORTRADAR_FIFA_RANKINGS_URL = "https://api.sportradar.com/soccer-extended/trial/v4/en/fifa_rankings.json"
LAST_N_MATCHES = 15
DEFAULT_LOOKBACK_DAYS = 900
RESULT_LABELS = {"H", "D", "A"}


def d(start, end):
    return (start, end)


UPCOMING_ESPN_COMPETITIONS = {
    "FIFA/World Cup": {"espn_id": "fifa.world", "priority": 1, "ranges": []},
    "FIFA/World Cup Qualifying - UEFA": {"espn_id": "fifa.worldq.uefa", "priority": 2, "ranges": []},
    "FIFA/World Cup Qualifying - CONMEBOL": {"espn_id": "fifa.worldq.conmebol", "priority": 3, "ranges": []},
    "FIFA/World Cup Qualifying - CONCACAF": {"espn_id": "fifa.worldq.concacaf", "priority": 4, "ranges": []},
    "FIFA/Friendly": {"espn_id": "fifa.friendly", "priority": 5, "ranges": []},
    "UEFA/European Championship": {"espn_id": "uefa.euro", "priority": 6, "ranges": []},
    "CONMEBOL/Copa America": {"espn_id": "conmebol.america", "priority": 7, "ranges": []},
    "CONCACAF/Gold Cup": {"espn_id": "concacaf.gold", "priority": 8, "ranges": []},
    "CAF/Africa Cup of Nations": {"espn_id": "caf.nations", "priority": 9, "ranges": []},
    "UEFA/Nations League": {"espn_id": "uefa.nations", "priority": 10, "ranges": []},
}

RECENT_ESPN_COMPETITIONS = {
    **UPCOMING_ESPN_COMPETITIONS,
    "FIFA/World Cup Qualifying - AFC": {"espn_id": "fifa.worldq.afc", "priority": 11, "ranges": []},
    "FIFA/World Cup Qualifying - CAF": {"espn_id": "fifa.worldq.caf", "priority": 12, "ranges": []},
    "AFC/Asian Cup": {"espn_id": "afc.cup", "priority": 13, "ranges": []},
}

FOOTBALL_DATA_COMPETITIONS = {
    "WC": "FIFA/World Cup",
    "EC": "UEFA/European Championship",
}


# This seed snapshot keeps the predictor usable without paid ranking/value API
# credentials. Refresh with --rankings-file/--squad-values-file when newer data is available.
DEFAULT_FIFA_RANKINGS = {
    "Argentina": {"rank": 1, "points": 1886.16},
    "Spain": {"rank": 2, "points": 1854.64},
    "France": {"rank": 3, "points": 1852.71},
    "England": {"rank": 4, "points": 1819.20},
    "Brazil": {"rank": 5, "points": 1776.03},
    "Portugal": {"rank": 6, "points": 1770.53},
    "Netherlands": {"rank": 7, "points": 1758.18},
    "Belgium": {"rank": 8, "points": 1736.38},
    "Italy": {"rank": 9, "points": 1717.15},
    "Germany": {"rank": 10, "points": 1716.98},
    "Croatia": {"rank": 11, "points": 1707.51},
    "Morocco": {"rank": 12, "points": 1694.24},
    "Uruguay": {"rank": 13, "points": 1680.75},
    "Colombia": {"rank": 14, "points": 1679.41},
    "Japan": {"rank": 15, "points": 1654.44},
    "United States": {"rank": 16, "points": 1645.48},
    "Mexico": {"rank": 17, "points": 1640.67},
    "Switzerland": {"rank": 18, "points": 1635.32},
    "Senegal": {"rank": 19, "points": 1630.32},
    "Denmark": {"rank": 20, "points": 1627.58},
    "Iran": {"rank": 21, "points": 1618.78},
    "Austria": {"rank": 22, "points": 1608.80},
    "South Korea": {"rank": 23, "points": 1585.45},
    "Australia": {"rank": 24, "points": 1571.29},
    "Turkey": {"rank": 25, "points": 1565.75},
    "Ecuador": {"rank": 26, "points": 1558.18},
    "Ukraine": {"rank": 27, "points": 1554.94},
    "Canada": {"rank": 28, "points": 1549.16},
    "Poland": {"rank": 29, "points": 1537.47},
    "Wales": {"rank": 30, "points": 1531.38},
    "Egypt": {"rank": 31, "points": 1526.25},
    "Serbia": {"rank": 32, "points": 1518.76},
    "Norway": {"rank": 33, "points": 1514.59},
    "Russia": {"rank": 34, "points": 1512.32},
    "Paraguay": {"rank": 35, "points": 1510.22},
    "Panama": {"rank": 36, "points": 1508.19},
    "Ivory Coast": {"rank": 37, "points": 1505.64},
    "Costa Rica": {"rank": 38, "points": 1504.38},
    "Scotland": {"rank": 39, "points": 1499.12},
    "Tunisia": {"rank": 40, "points": 1497.41},
    "Algeria": {"rank": 41, "points": 1492.83},
    "Czechia": {"rank": 42, "points": 1489.53},
    "Nigeria": {"rank": 43, "points": 1488.86},
    "Slovakia": {"rank": 44, "points": 1484.19},
    "Greece": {"rank": 45, "points": 1483.95},
    "Romania": {"rank": 46, "points": 1481.40},
    "Hungary": {"rank": 47, "points": 1478.73},
    "Cameroon": {"rank": 48, "points": 1474.74},
    "Chile": {"rank": 49, "points": 1473.58},
    "Venezuela": {"rank": 50, "points": 1471.53},
    "Qatar": {"rank": 51, "points": 1469.12},
    "Saudi Arabia": {"rank": 58, "points": 1409.92},
    "South Africa": {"rank": 59, "points": 1407.91},
    "Ghana": {"rank": 60, "points": 1404.77},
    "Jamaica": {"rank": 62, "points": 1399.12},
    "Haiti": {"rank": 83, "points": 1287.75},
    "New Zealand": {"rank": 89, "points": 1260.32},
    "Curacao": {"rank": 90, "points": 1258.71},
    "Cape Verde": {"rank": 73, "points": 1338.41},
    "Uzbekistan": {"rank": 57, "points": 1412.52},
    "Jordan": {"rank": 64, "points": 1383.09},
    "Bosnia-Herzegovina": {"rank": 74, "points": 1335.60},
}

DEFAULT_SQUAD_VALUES_EUR_M = {
    "England": 1520.0,
    "Brazil": 1260.0,
    "France": 1230.0,
    "Portugal": 1050.0,
    "Spain": 965.5,
    "Germany": 831.0,
    "Netherlands": 815.0,
    "Argentina": 805.0,
    "Italy": 705.5,
    "Belgium": 560.0,
    "Denmark": 420.0,
    "Uruguay": 410.0,
    "Morocco": 385.0,
    "United States": 360.0,
    "Colombia": 340.0,
    "Croatia": 320.0,
    "Mexico": 295.0,
    "Japan": 285.0,
    "Switzerland": 280.0,
    "Turkey": 270.0,
    "Serbia": 265.0,
    "Ecuador": 240.0,
    "Canada": 230.0,
    "Poland": 225.0,
    "South Korea": 200.0,
    "Austria": 195.0,
    "Senegal": 190.0,
    "Ukraine": 185.0,
    "Czechia": 170.0,
    "Norway": 165.0,
    "Australia": 155.0,
    "Iran": 145.0,
    "Egypt": 140.0,
    "Ghana": 135.0,
    "Nigeria": 130.0,
    "Scotland": 125.0,
    "Paraguay": 120.0,
    "Ivory Coast": 115.0,
    "Qatar": 80.0,
    "Saudi Arabia": 72.0,
    "South Africa": 70.0,
    "Uzbekistan": 68.0,
    "Jordan": 45.0,
    "Cape Verde": 42.0,
    "Bosnia-Herzegovina": 40.0,
    "Haiti": 30.0,
    "Curacao": 28.0,
    "New Zealand": 27.0,
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Build a current-context national-team predictor from rankings, squad values, and last-15 games."
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing recent-match CSV/context files.")
    parser.add_argument(
        "--world-cup-only",
        action="store_true",
        help="Build context for current World Cup teams only. This is the default pipeline use.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="How many days of ESPN national-team scoreboards to scan for last-15 matches.",
    )
    parser.add_argument(
        "--rankings-file",
        default=FIFA_RANKINGS_FILE,
        help="JSON/CSV file with current FIFA rankings. JSON can be {team:{rank,points}} or a list of rows.",
    )
    parser.add_argument(
        "--squad-values-file",
        default=SQUAD_VALUES_FILE,
        help="JSON/CSV file with latest national squad market values in EUR millions.",
    )
    parser.add_argument(
        "--footballdata-io-token",
        default=os.getenv("FOOTBALLDATA_IO_TOKEN", "").strip(),
        help="Optional Footballdata.io token for current FIFA ranking refresh.",
    )
    parser.add_argument(
        "--sportradar-api-key",
        default=os.getenv("SPORTRADAR_API_KEY", "").strip(),
        help="Optional Sportradar API key for current FIFA ranking refresh.",
    )
    return parser.parse_args()


def normalize_team_key(name):
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    parts = [part for part in text.split() if part and part not in {"fc", "cf", "team", "national", "football", "soccer", "the"}]
    aliases = {
        "usa": "unitedstates",
        "us": "unitedstates",
        "u s": "unitedstates",
        "unitedstatesofamerica": "unitedstates",
        "czechrepublic": "czechia",
        "turkiye": "turkey",
        "ivorycoast": "cotedivoire",
        "cotedivoire": "cotedivoire",
        "bosniaherzegovina": "bosniaandherzegovina",
        "bosniaherz": "bosniaandherzegovina",
        "curacao": "curacao",
        "korea republic": "southkorea",
        "republickorea": "southkorea",
    }
    key = "".join(parts)
    return aliases.get(key, key)


def canonical_team_name(name):
    text = str(name or "").strip()
    aliases = {
        "USA": "United States",
        "US": "United States",
        "Czech Republic": "Czechia",
        "Türkiye": "Turkey",
        "Korea Republic": "South Korea",
        "Bosnia and Herzegovina": "Bosnia-Herzegovina",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
    }
    return aliases.get(text, text)


def make_prediction_key(match_date, competition, home_team, away_team):
    parsed = pd.to_datetime(match_date, utc=True, errors="coerce")
    date_part = parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else str(match_date)[:10]
    pair = sorted([normalize_team_key(home_team), normalize_team_key(away_team)])
    return f"{date_part}|{competition}|{pair[0]}|{pair[1]}"


def fetch_json(url, headers=None, timeout=30):
    request = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


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

    home_team = canonical_team_name(((home.get("team") or {}).get("displayName") or "").strip())
    away_team = canonical_team_name(((away.get("team") or {}).get("displayName") or "").strip())
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

    stage = str((event.get("season") or {}).get("slug", "") or "").strip().lower() or "unknown"
    venue = competition.get("venue") or {}
    return {
        "match_id": str(event.get("id", "")).strip(),
        "match_datetime_utc": event_dt.isoformat(),
        "match_date": event_dt.strftime("%Y-%m-%d"),
        "competition": competition_name,
        "stage": stage,
        "home_team": home_team,
        "away_team": away_team,
        "FTHG": int(home_goals) if home_goals is not None else None,
        "FTAG": int(away_goals) if away_goals is not None else None,
        "FTR": result,
        "status": str(status_type.get("name", "") or status_type.get("description", "")).strip(),
        "is_neutral_site": bool(competition.get("neutralSite")) if competition.get("neutralSite") is not None else True,
        "venue": str(venue.get("fullName", "") or venue.get("name", "") or "").strip(),
        "source": "espn",
    }


def parse_football_data_match(match, competition_name, completed_only=True):
    status = str(match.get("status", "")).strip().upper()
    if completed_only and status not in {"FINISHED", "FULL_TIME"}:
        return None
    home_team = canonical_team_name(((match.get("homeTeam") or {}).get("name") or "").strip())
    away_team = canonical_team_name(((match.get("awayTeam") or {}).get("name") or "").strip())
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
        "status": status,
        "is_neutral_site": True,
        "venue": str(match.get("venue", "") or "").strip(),
        "source": "football-data.org",
    }


def fetch_world_cup_team_names(start_date="2026-06-11", end_date="2026-06-27"):
    teams = set()
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    for day in date_range(start, end):
        url = ESPN_SCOREBOARD_API.format(espn_id="fifa.world") + f"?dates={day.strftime('%Y%m%d')}"
        try:
            payload = fetch_json(url, timeout=30)
        except Exception:
            continue
        for event in payload.get("events") or []:
            parsed = parse_espn_event(event, "FIFA/World Cup", require_completed=False)
            if parsed:
                for side in ["home_team", "away_team"]:
                    name = parsed.get(side, "")
                    if name and not is_placeholder_team(name):
                        teams.add(name)
    return sorted(teams)


def is_placeholder_team(name):
    text = str(name or "").lower()
    return any(token in text for token in ["group ", "winner", "third place", "round of", "quarterfinal", "semifinal"])


def fetch_recent_espn_matches(target_teams, lookback_days):
    today = datetime.now(UTC).date()
    start = today - timedelta(days=max(30, int(lookback_days)))
    target_keys = {normalize_team_key(team) for team in target_teams}
    by_team = {team: [] for team in target_teams}
    all_rows = []
    seen = set()

    for competition_name, config in sorted(RECENT_ESPN_COMPETITIONS.items(), key=lambda item: item[1]["priority"]):
        espn_id = config["espn_id"]
        print(f"Scanning ESPN {competition_name} for recent national-team matches...")
        for day in date_range(start, today):
            if all(len(matches) >= LAST_N_MATCHES for matches in by_team.values()):
                break
            url = ESPN_SCOREBOARD_API.format(espn_id=espn_id) + f"?dates={day.strftime('%Y%m%d')}"
            try:
                payload = fetch_json(url, timeout=30)
            except Exception:
                continue
            for event in payload.get("events") or []:
                event_id = str(event.get("id", "")).strip()
                if event_id and event_id in seen:
                    continue
                parsed = parse_espn_event(event, competition_name, require_completed=True)
                if not parsed or str(parsed.get("FTR", "")).strip() not in RESULT_LABELS:
                    continue
                home_key = normalize_team_key(parsed["home_team"])
                away_key = normalize_team_key(parsed["away_team"])
                if home_key not in target_keys and away_key not in target_keys:
                    continue
                if event_id:
                    seen.add(event_id)
                all_rows.append(parsed)
                for team in target_teams:
                    key = normalize_team_key(team)
                    if key in {home_key, away_key} and len(by_team[team]) < LAST_N_MATCHES:
                        by_team[team].append(parsed)
    all_rows.sort(key=lambda row: row["match_datetime_utc"])
    return all_rows, by_team


def load_json_or_csv_records(path):
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".csv"):
        return pd.read_csv(path).to_dict("records")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_rankings_payload(payload):
    rankings = {}
    if isinstance(payload, dict):
        if "rankings" in payload:
            for ranking_group in payload.get("rankings") or []:
                for row in ranking_group.get("competitor_rankings") or []:
                    competitor = row.get("competitor") or {}
                    name = canonical_team_name(competitor.get("name") or competitor.get("country") or "")
                    if name:
                        rankings[name] = {"rank": int(row.get("rank", 999)), "points": float(row.get("points", 0.0) or 0.0)}
            return rankings
        for team, value in payload.items():
            team_name = canonical_team_name(team)
            if isinstance(value, dict):
                rankings[team_name] = {
                    "rank": int(value.get("rank", value.get("position", 999)) or 999),
                    "points": float(value.get("points", value.get("rating", 0.0)) or 0.0),
                }
            else:
                rankings[team_name] = {"rank": int(value or 999), "points": 0.0}
        return rankings
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = canonical_team_name(row.get("team") or row.get("country") or row.get("name") or "")
            if not name:
                continue
            rankings[name] = {
                "rank": int(row.get("rank", row.get("position", 999)) or 999),
                "points": float(row.get("points", row.get("rating", 0.0)) or 0.0),
            }
    return rankings


def fetch_rankings_from_footballdata_io(token):
    if not token:
        return {}
    url = f"{FOOTBALLDATA_IO_BASE}/fifa-rankings/current?ranking_type=men"
    try:
        payload = fetch_json(url, headers={"Authorization": f"Bearer {token}"}, timeout=45)
    except Exception as exc:
        print(f"Footballdata.io FIFA rankings fetch failed: {exc}")
        return {}
    return normalize_rankings_payload(payload)


def fetch_rankings_from_sportradar(api_key):
    if not api_key:
        return {}
    query = urllib.parse.urlencode({"api_key": api_key})
    try:
        payload = fetch_json(f"{SPORTRADAR_FIFA_RANKINGS_URL}?{query}", timeout=45)
    except Exception as exc:
        print(f"Sportradar FIFA rankings fetch failed: {exc}")
        return {}
    return normalize_rankings_payload(payload)


def load_fifa_rankings(path, footballdata_io_token="", sportradar_api_key=""):
    rankings = dict(DEFAULT_FIFA_RANKINGS)
    payload = load_json_or_csv_records(path)
    if payload:
        rankings.update(normalize_rankings_payload(payload))
    api_rankings = fetch_rankings_from_footballdata_io(footballdata_io_token)
    if not api_rankings:
        api_rankings = fetch_rankings_from_sportradar(sportradar_api_key)
    if api_rankings:
        rankings.update(api_rankings)
    normalized = {}
    for team, row in rankings.items():
        normalized[canonical_team_name(team)] = {"rank": int(row.get("rank", 999)), "points": float(row.get("points", 0.0) or 0.0)}
    return normalized


def parse_value_to_eur_m(value):
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    multiplier = 1.0
    lowered = text.lower()
    if "bn" in lowered or "billion" in lowered or "b" in lowered:
        multiplier = 1000.0
    elif "k" in lowered:
        multiplier = 0.001
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0)) * multiplier


def normalize_squad_values_payload(payload):
    values = {}
    if isinstance(payload, dict):
        for team, value in payload.items():
            team_name = canonical_team_name(team)
            if isinstance(value, dict):
                raw = value.get("squad_value_eur_m", value.get("market_value_eur_m", value.get("value", 0.0)))
                updated = value.get("updated_at", "")
            else:
                raw = value
                updated = ""
            values[team_name] = {"squad_value_eur_m": parse_value_to_eur_m(raw), "updated_at": str(updated)}
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = canonical_team_name(row.get("team") or row.get("country") or row.get("name") or "")
            if not name:
                continue
            raw = row.get("squad_value_eur_m", row.get("market_value_eur_m", row.get("value", row.get("market_value", 0.0))))
            values[name] = {"squad_value_eur_m": parse_value_to_eur_m(raw), "updated_at": str(row.get("updated_at", ""))}
    return values


def load_squad_values(path):
    values = {
        team: {"squad_value_eur_m": float(value), "updated_at": "seed"}
        for team, value in DEFAULT_SQUAD_VALUES_EUR_M.items()
    }
    payload = load_json_or_csv_records(path)
    if payload:
        values.update(normalize_squad_values_payload(payload))
    return {canonical_team_name(team): value for team, value in values.items()}


def result_for_team(row, team):
    is_home = normalize_team_key(row["home_team"]) == normalize_team_key(team)
    gf = int(row["FTHG"] if is_home else row["FTAG"])
    ga = int(row["FTAG"] if is_home else row["FTHG"])
    if gf > ga:
        result = "W"
        points = 3
    elif gf < ga:
        result = "L"
        points = 0
    else:
        result = "D"
        points = 1
    return result, points, gf, ga


def summarize_last_matches(team, matches):
    ordered = sorted(matches, key=lambda row: row["match_datetime_utc"], reverse=True)[:LAST_N_MATCHES]
    if not ordered:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "points_per_game": 0.0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "goals_for_per_game": 0.0,
            "goals_against_per_game": 0.0,
            "clean_sheets": 0,
            "failed_to_score": 0,
            "form": "",
            "matches": [],
        }
    wins = draws = losses = points = gf_total = ga_total = clean_sheets = failed_to_score = 0
    form = []
    compact_matches = []
    for row in ordered:
        result, pts, gf, ga = result_for_team(row, team)
        wins += 1 if result == "W" else 0
        draws += 1 if result == "D" else 0
        losses += 1 if result == "L" else 0
        points += pts
        gf_total += gf
        ga_total += ga
        clean_sheets += 1 if ga == 0 else 0
        failed_to_score += 1 if gf == 0 else 0
        form.append(result)
        compact_matches.append(
            {
                "date": row["match_date"],
                "competition": row["competition"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": int(row["FTHG"]),
                "away_goals": int(row["FTAG"]),
                "team_result": result,
            }
        )
    games = len(ordered)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "points_per_game": round(points / games, 4),
        "goals_for": gf_total,
        "goals_against": ga_total,
        "goal_diff": gf_total - ga_total,
        "goals_for_per_game": round(gf_total / games, 4),
        "goals_against_per_game": round(ga_total / games, 4),
        "clean_sheets": clean_sheets,
        "failed_to_score": failed_to_score,
        "form": "".join(form),
        "matches": compact_matches,
    }


def build_team_context(target_teams, by_team_matches, rankings, squad_values):
    context = {}
    for team in target_teams:
        team_name = canonical_team_name(team)
        ranking = rankings.get(team_name, {"rank": 999, "points": 0.0})
        squad = squad_values.get(team_name, {"squad_value_eur_m": 0.0, "updated_at": ""})
        last15 = summarize_last_matches(team_name, by_team_matches.get(team, []))
        context[team_name] = {
            "fifa_rank": int(ranking.get("rank", 999) or 999),
            "fifa_points": float(ranking.get("points", 0.0) or 0.0),
            "squad_value_eur_m": float(squad.get("squad_value_eur_m", 0.0) or 0.0),
            "squad_value_updated_at": str(squad.get("updated_at", "")),
            "last15": last15,
        }
    return context


def load_existing_recent_matches():
    if not os.path.exists(RAW_MATCHES_FILE):
        return [], {}
    frame = pd.read_csv(RAW_MATCHES_FILE)
    rows = frame.to_dict("records")
    by_team = defaultdict(list)
    for row in rows:
        for side in ["home_team", "away_team"]:
            by_team[canonical_team_name(row.get(side, ""))].append(row)
    return rows, by_team


def strength_from_context(team_context):
    rank = float(team_context.get("fifa_rank", 999) or 999)
    fifa_points = float(team_context.get("fifa_points", 0.0) or 0.0)
    value = float(team_context.get("squad_value_eur_m", 0.0) or 0.0)
    form = team_context.get("last15", {})
    ppg = float(form.get("points_per_game", 0.0) or 0.0)
    gdpg = safe_div(form.get("goal_diff", 0.0), max(1, form.get("games", 0)))
    gfpg = float(form.get("goals_for_per_game", 0.0) or 0.0)
    gapg = float(form.get("goals_against_per_game", 0.0) or 0.0)
    ranking_score = max(0.0, min(1.0, (210.0 - rank) / 209.0))
    points_score = max(0.0, min(1.0, (fifa_points - 900.0) / 1000.0)) if fifa_points else ranking_score
    value_score = max(0.0, min(1.0, math.log1p(value) / math.log1p(1600.0))) if value > 0 else 0.35
    form_score = max(0.0, min(1.0, ppg / 3.0))
    goal_score = max(0.0, min(1.0, 0.50 + gdpg / 5.0 + (gfpg - gapg) / 8.0))
    return (
        0.34 * ranking_score
        + 0.16 * points_score
        + 0.20 * value_score
        + 0.22 * form_score
        + 0.08 * goal_score
    )


def safe_div(numerator, denominator, default=0.0):
    try:
        denominator = float(denominator)
        if denominator == 0:
            return default
        return float(numerator) / denominator
    except Exception:
        return default


def build_feature_row(home_team, away_team, competition, stage, is_neutral_site, team_context):
    home = team_context.get(canonical_team_name(home_team), {})
    away = team_context.get(canonical_team_name(away_team), {})
    h_form = home.get("last15", {})
    a_form = away.get("last15", {})
    h_strength = strength_from_context(home)
    a_strength = strength_from_context(away)
    return {
        "home_strength": h_strength,
        "away_strength": a_strength,
        "strength_diff": h_strength - a_strength,
        "home_fifa_rank": float(home.get("fifa_rank", 999) or 999),
        "away_fifa_rank": float(away.get("fifa_rank", 999) or 999),
        "fifa_rank_diff": float(away.get("fifa_rank", 999) or 999) - float(home.get("fifa_rank", 999) or 999),
        "home_fifa_points": float(home.get("fifa_points", 0.0) or 0.0),
        "away_fifa_points": float(away.get("fifa_points", 0.0) or 0.0),
        "fifa_points_diff": float(home.get("fifa_points", 0.0) or 0.0) - float(away.get("fifa_points", 0.0) or 0.0),
        "home_squad_value_eur_m": float(home.get("squad_value_eur_m", 0.0) or 0.0),
        "away_squad_value_eur_m": float(away.get("squad_value_eur_m", 0.0) or 0.0),
        "squad_value_ratio": safe_div(float(home.get("squad_value_eur_m", 0.0) or 0.0) + 25.0, float(away.get("squad_value_eur_m", 0.0) or 0.0) + 25.0, 1.0),
        "home_last15_ppg": float(h_form.get("points_per_game", 0.0) or 0.0),
        "away_last15_ppg": float(a_form.get("points_per_game", 0.0) or 0.0),
        "last15_ppg_diff": float(h_form.get("points_per_game", 0.0) or 0.0) - float(a_form.get("points_per_game", 0.0) or 0.0),
        "home_last15_gfpg": float(h_form.get("goals_for_per_game", 0.0) or 0.0),
        "away_last15_gfpg": float(a_form.get("goals_for_per_game", 0.0) or 0.0),
        "home_last15_gapg": float(h_form.get("goals_against_per_game", 0.0) or 0.0),
        "away_last15_gapg": float(a_form.get("goals_against_per_game", 0.0) or 0.0),
        "last15_goal_diff_delta": safe_div(h_form.get("goal_diff", 0.0), max(1, h_form.get("games", 0))) - safe_div(a_form.get("goal_diff", 0.0), max(1, a_form.get("games", 0))),
        "is_world_cup": 1.0 if "world cup" in str(competition).lower() else 0.0,
        "is_knockout": 1.0 if stage_is_knockout(stage) else 0.0,
        "is_neutral_site": 1.0 if is_neutral_site else 0.0,
        "competition": competition,
        "stage": str(stage or "unknown").strip().lower() or "unknown",
    }


def stage_is_knockout(stage):
    text = str(stage or "").lower()
    if "group" in text or "qualifying" in text:
        return False
    return any(term in text for term in ["final", "semi", "quarter", "round", "knockout", "third"])


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
        snapshot.get("team_context", {}),
    )
    return pd.DataFrame([row]), home, away


def resolve_team_name(raw_name, known_teams):
    raw = canonical_team_name(str(raw_name or "").strip())
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


def align_feature_frame(raw_feature_frame, bundle):
    return raw_feature_frame.reindex(columns=bundle.get("train_columns", list(raw_feature_frame.columns)), fill_value=0.0)


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


def context_probabilities(home_context, away_context, is_neutral_site=True):
    h_strength = strength_from_context(home_context)
    a_strength = strength_from_context(away_context)
    diff = h_strength - a_strength
    rank_gap = float(away_context.get("fifa_rank", 999) or 999) - float(home_context.get("fifa_rank", 999) or 999)
    value_ratio = safe_div(
        float(home_context.get("squad_value_eur_m", 0.0) or 0.0) + 25.0,
        float(away_context.get("squad_value_eur_m", 0.0) or 0.0) + 25.0,
        1.0,
    )
    logistic_input = (4.4 * diff) + (0.006 * rank_gap) + (0.16 * math.log(max(0.05, value_ratio)))
    if not is_neutral_site:
        logistic_input += 0.12
    home_away_split = 1.0 / (1.0 + math.exp(-logistic_input))
    strength_gap = abs(diff)
    draw = max(0.16, min(0.31, 0.29 - 0.20 * strength_gap))
    non_draw = 1.0 - draw
    home = non_draw * home_away_split
    away = non_draw * (1.0 - home_away_split)
    return {"H": home, "D": draw, "A": away}


def expected_goals(home_context, away_context):
    h_form = home_context.get("last15", {})
    a_form = away_context.get("last15", {})
    h_gf = float(h_form.get("goals_for_per_game", 1.15) or 1.15)
    h_ga = float(h_form.get("goals_against_per_game", 1.15) or 1.15)
    a_gf = float(a_form.get("goals_for_per_game", 1.15) or 1.15)
    a_ga = float(a_form.get("goals_against_per_game", 1.15) or 1.15)
    h_strength = strength_from_context(home_context)
    a_strength = strength_from_context(away_context)
    home_goals = max(0.25, min(3.75, 0.56 * h_gf + 0.44 * a_ga + 0.55 * (h_strength - a_strength)))
    away_goals = max(0.25, min(3.75, 0.56 * a_gf + 0.44 * h_ga + 0.55 * (a_strength - h_strength)))
    return home_goals, away_goals


class CurrentContextClassifier:
    def __init__(self, team_context=None):
        self.team_context = team_context or {}
        self.classes_ = [0, 1, 2]

    def predict_proba(self, rows):
        matrices = []
        for _, row in rows.iterrows():
            probs = self._probabilities_from_feature_row(row)
            matrices.append([probs["A"], probs["D"], probs["H"]])
        return matrices

    def predict(self, rows):
        return [max(range(3), key=lambda idx: probs[idx]) for probs in self.predict_proba(rows)]

    def _probabilities_from_feature_row(self, row):
        diff = float(row.get("strength_diff", 0.0) or 0.0)
        rank_gap = float(row.get("fifa_rank_diff", 0.0) or 0.0)
        value_ratio = float(row.get("squad_value_ratio", 1.0) or 1.0)
        logistic_input = (4.4 * diff) + (0.006 * rank_gap) + (0.16 * math.log(max(0.05, value_ratio)))
        home_away_split = 1.0 / (1.0 + math.exp(-logistic_input))
        draw = max(0.16, min(0.31, 0.29 - 0.20 * abs(diff)))
        non_draw = 1.0 - draw
        return {"H": non_draw * home_away_split, "D": draw, "A": non_draw * (1.0 - home_away_split)}


class ContextGoalRegressor:
    def __init__(self, side):
        self.side = side

    def predict(self, rows):
        out = []
        for _, row in rows.iterrows():
            if self.side == "home":
                gf = float(row.get("home_last15_gfpg", 1.15) or 1.15)
                opp_ga = float(row.get("away_last15_gapg", 1.15) or 1.15)
                diff = float(row.get("strength_diff", 0.0) or 0.0)
                out.append(max(0.25, min(3.75, 0.56 * gf + 0.44 * opp_ga + 0.55 * diff)))
            else:
                gf = float(row.get("away_last15_gfpg", 1.15) or 1.15)
                opp_ga = float(row.get("home_last15_gapg", 1.15) or 1.15)
                diff = -float(row.get("strength_diff", 0.0) or 0.0)
                out.append(max(0.25, min(3.75, 0.56 * gf + 0.44 * opp_ga + 0.55 * diff)))
        return out


class ResultLabelEncoder:
    def inverse_transform(self, values):
        mapping = {0: "A", 1: "D", 2: "H", "0": "A", "1": "D", "2": "H"}
        return [mapping.get(value, value) for value in values]


CurrentContextClassifier.__module__ = "Process_National_Team_Data"
ContextGoalRegressor.__module__ = "Process_National_Team_Data"
ResultLabelEncoder.__module__ = "Process_National_Team_Data"


def build_context_bundle(team_context, recent_rows, target_teams):
    sample_rows = []
    teams = list(target_teams)
    if len(teams) >= 2:
        sample_rows.append(
            build_feature_row(teams[0], teams[1], "FIFA/World Cup", "group-stage", True, team_context)
        )
    train_columns = list(pd.DataFrame(sample_rows or [{}]).columns)
    snapshot = {
        "team_context": team_context,
        "known_teams": sorted(team_context.keys()),
        "recent_matches": recent_rows,
        "latest_match_datetime_utc": max((row.get("match_datetime_utc", "") for row in recent_rows), default=""),
    }
    return {
        "model_version": 2,
        "model_type": "current_context",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "fingerprint": hashlib.sha256(json.dumps(team_context, sort_keys=True).encode("utf-8")).hexdigest(),
        "clf": CurrentContextClassifier(team_context),
        "result_label_encoder": ResultLabelEncoder(),
        "home_goal_reg": ContextGoalRegressor("home"),
        "away_goal_reg": ContextGoalRegressor("away"),
        "train_columns": train_columns,
        "categorical_feature_columns": [],
        "snapshot": snapshot,
        "training_rows": 0,
        "data_basis": "current FIFA rankings, latest squad market values, ESPN last-15 matches across competitions and friendlies",
    }


def load_model_bundle(path=MODEL_CACHE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"National context cache not found at {path}. Run Process_National_Team_Data.py first."
        )
    return joblib.load(path)


def write_api_report(target_teams, ranking_source, value_source, recent_match_count):
    report = {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "summary": [
            "National predictions no longer train from previous World Cup tournament history.",
            "ESPN scoreboard APIs provide last-15 match background across World Cup, qualifiers, continental tournaments, Nations League, and friendlies.",
            "Current FIFA rankings are loaded from a provided file or optional paid API credentials; seed rankings are used when no refresh source is configured.",
            "Latest squad Transfermarkt-style market values are loaded from a provided JSON/CSV snapshot; seed values are used when no refresh source is configured.",
        ],
        "target_team_count": len(target_teams),
        "recent_match_count": recent_match_count,
        "ranking_source": ranking_source,
        "squad_value_source": value_source,
        "outputs": {
            "raw_recent_matches": RAW_MATCHES_FILE,
            "processed_context": PROCESSED_MATCHES_FILE,
            "model_cache": MODEL_CACHE,
        },
    }
    os.makedirs(NATIONAL_DATA_DIR, exist_ok=True)
    with open(API_REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def run_pipeline(args):
    os.makedirs(NATIONAL_DATA_DIR, exist_ok=True)
    target_teams = fetch_world_cup_team_names() if args.world_cup_only else fetch_world_cup_team_names()
    if not target_teams:
        target_teams = sorted(DEFAULT_FIFA_RANKINGS.keys())

    rankings = load_fifa_rankings(
        args.rankings_file,
        footballdata_io_token=getattr(args, "footballdata_io_token", ""),
        sportradar_api_key=getattr(args, "sportradar_api_key", ""),
    )
    squad_values = load_squad_values(args.squad_values_file)

    if args.skip_fetch:
        recent_rows, by_team = load_existing_recent_matches()
    else:
        recent_rows, by_team = fetch_recent_espn_matches(target_teams, args.lookback_days)
        pd.DataFrame(recent_rows).to_csv(RAW_MATCHES_FILE, index=False)

    team_context = build_team_context(target_teams, by_team, rankings, squad_values)
    processed_rows = []
    for team, context in team_context.items():
        row = {
            "team": team,
            "fifa_rank": context["fifa_rank"],
            "fifa_points": context["fifa_points"],
            "squad_value_eur_m": context["squad_value_eur_m"],
            **{f"last15_{key}": value for key, value in context["last15"].items() if key != "matches"},
        }
        processed_rows.append(row)
    pd.DataFrame(processed_rows).to_csv(PROCESSED_MATCHES_FILE, index=False)

    bundle = build_context_bundle(team_context, recent_rows, target_teams)
    joblib.dump(bundle, MODEL_CACHE)
    ranking_source = "api_or_file" if os.path.exists(args.rankings_file) or getattr(args, "footballdata_io_token", "") or getattr(args, "sportradar_api_key", "") else "seed_snapshot"
    value_source = "file" if os.path.exists(args.squad_values_file) else "seed_snapshot"
    write_api_report(target_teams, ranking_source, value_source, len(recent_rows))

    print("\nNational-team current-context predictor generated")
    print(f"Teams loaded: {len(target_teams)}")
    print(f"Recent ESPN matches loaded: {len(recent_rows)}")
    print(f"Ranking source: {ranking_source}")
    print(f"Squad value source: {value_source}")
    print(f"Context data: {PROCESSED_MATCHES_FILE}")
    print(f"Model cache: {MODEL_CACHE}")
    return bundle


def main():
    args = parse_cli_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
