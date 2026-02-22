import argparse
import difflib
import json
import os
import urllib.request
import unicodedata
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL_PREDICTIONS_FILE = os.path.join(BASE_DIR, "Data", "Predictions", "upcoming_matchweek_predictions.csv")
GLOBAL_MAPPING_FILE = os.path.join(BASE_DIR, "Data", "Predictions", "upcoming_fixture_team_mapping.json")
MLS_PREDICTIONS_FILE = os.path.join(BASE_DIR, "MLS", "Data", "Predictions", "upcoming_matchweek_predictions.csv")
MLS_MAPPING_FILE = os.path.join(BASE_DIR, "MLS", "Data", "Predictions", "upcoming_fixture_team_mapping.json")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_COMPETITION_KEYS = {
    "England/Premier League": "eng.1",
    "England/Championship": "eng.2",
    "Spain/La Liga": "esp.1",
    "Spain/La Liga 2": "esp.2",
    "Italy/Serie A": "ita.1",
    "Italy/Serie B": "ita.2",
    "Germany/Bundesliga": "ger.1",
    "Germany/Bundesliga 2": "ger.2",
    "France/Ligue 1": "fra.1",
    "France/Ligue 2": "fra.2",
    "Portugal/Liga Portugal": "por.1",
    "United States/MLS": "usa.1",
}
MLS_COMPETITION = "United States/MLS"
EASTERN_TZ = ZoneInfo("America/New_York")

REQUIRED_COLUMNS = [
    "prediction_key",
    "created_at_utc",
    "match_date",
    "competition",
    "home_team",
    "away_team",
    "predicted_result",
    "actual_home_goals",
    "actual_away_goals",
    "actual_result",
    "is_correct",
    "settled_at_utc",
]
TEAM_KEY_ALIASES = {
    "lafc": "losangelesfc",
    "lagalaxy": "losangelesgalaxy",
    "stlouiscity": "stlouiscitysc",
    "stlouiscitysc": "stlouiscitysc",
    "dcunited": "dcunited",
    "newyorkcityfc": "newyorkcity",
    "newyorkredbulls": "newyorkredbulls",
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Update upcoming predictions with live/final real results and correctness flags."
    )
    parser.add_argument(
        "--cleanup-all",
        action="store_true",
        help="Clear all settled result fields for both European and MLS prediction CSVs.",
    )
    parser.add_argument(
        "--resettle-after-cleanup",
        action="store_true",
        help="When used with --cleanup-all, immediately resettle from ESPN in the same run.",
    )
    parser.add_argument(
        "--cleanup-mls",
        action="store_true",
        help="One-time cleanup: clear MLS settled rows that do not match completed ESPN results.",
    )
    return parser.parse_args()


def normalize_team_key(name):
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = text.replace("&", " and ")
    for ch in ("'", ".", "-", "_", "/", ","):
        text = text.replace(ch, " ")
    parts = [p for p in text.split() if p]
    stop_words = {
        "fc",
        "cf",
        "ac",
        "ca",
        "afc",
        "us",
        "sc",
        "sv",
        "fk",
        "the",
        "club",
        "de",
        "calcio",
        "team",
        "football",
    }
    parts = [p for p in parts if p not in stop_words]
    key = "".join(parts)
    return TEAM_KEY_ALIASES.get(key, key)


def fetch_json(url, headers=None, timeout=30):
    request = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_mapping(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_mapping(path, mapping):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    normalized = {}
    for competition, values in (mapping or {}).items():
        if not isinstance(values, dict):
            continue
        normalized[str(competition)] = dict(
            sorted(
                ((str(k).strip(), str(v).strip()) for k, v in values.items() if str(k).strip()),
                key=lambda item: item[0].lower(),
            )
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(normalized, fh, indent=2, ensure_ascii=False)


def load_predictions(path):
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    # Settlement columns must accept mixed text/number values.
    object_cols = [
        "prediction_key",
        "match_date",
        "competition",
        "home_team",
        "away_team",
        "predicted_result",
        "actual_result",
        "is_correct",
        "settled_at_utc",
    ]
    for col in object_cols:
        frame[col] = frame[col].astype("object")
    return frame


def infer_result_code(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def choose_score(score_obj):
    if not isinstance(score_obj, dict):
        return None, None
    for bucket in ("fullTime", "regularTime", "halfTime"):
        section = score_obj.get(bucket)
        if not isinstance(section, dict):
            continue
        hg = pd.to_numeric(section.get("home"), errors="coerce")
        ag = pd.to_numeric(section.get("away"), errors="coerce")
        if pd.notna(hg) and pd.notna(ag):
            return int(hg), int(ag)
    return None, None


def event_date_key_for_competition(dt_utc, competition):
    if competition == MLS_COMPETITION:
        return dt_utc.tz_convert(EASTERN_TZ).strftime("%Y-%m-%d")
    return dt_utc.tz_convert("UTC").strftime("%Y-%m-%d")


def update_frame_with_results(frame, results_index):
    if frame is None or frame.empty:
        return frame, 0
    updates = 0
    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
    for idx, row in frame.iterrows():
        date_key = str(row.get("match_date", "")).strip()
        comp_key = str(row.get("competition", "")).strip()
        home_key = normalize_team_key(row.get("home_team", ""))
        away_key = normalize_team_key(row.get("away_team", ""))
        if not date_key or not comp_key or not home_key or not away_key:
            continue
        key = (date_key, comp_key, home_key, away_key)
        result = results_index.get(key)
        if result is None:
            key_rev = (date_key, comp_key, away_key, home_key)
            result = results_index.get(key_rev)
            if result is None:
                continue
            result = {
                "actual_home_goals": result["actual_away_goals"],
                "actual_away_goals": result["actual_home_goals"],
                "actual_result": "A" if result["actual_result"] == "H" else "H" if result["actual_result"] == "A" else "D",
                "completed": bool(result.get("completed")),
            }
        if not bool(result.get("completed")):
            continue

        prev_hg = pd.to_numeric(row.get("actual_home_goals"), errors="coerce")
        prev_ag = pd.to_numeric(row.get("actual_away_goals"), errors="coerce")
        prev_res = str(row.get("actual_result", "")).strip().upper()
        prev_correct = str(row.get("is_correct", "")).strip()
        new_correct = "1" if str(row.get("predicted_result", "")).strip().upper() == result["actual_result"] else "0"

        changed = (
            pd.isna(prev_hg) or int(prev_hg) != int(result["actual_home_goals"]) or
            pd.isna(prev_ag) or int(prev_ag) != int(result["actual_away_goals"]) or
            prev_res != result["actual_result"] or
            prev_correct != new_correct
        )
        if not changed:
            continue

        frame.at[idx, "actual_home_goals"] = result["actual_home_goals"]
        frame.at[idx, "actual_away_goals"] = result["actual_away_goals"]
        frame.at[idx, "actual_result"] = result["actual_result"]
        frame.at[idx, "is_correct"] = new_correct
        frame.at[idx, "settled_at_utc"] = now_utc
        updates += 1
    return frame, updates


def clear_future_settled_rows(frame, today_date):
    if frame is None or frame.empty:
        return frame, 0
    cleared = 0
    for idx, row in frame.iterrows():
        existing_result = str(row.get("actual_result", "")).strip().upper()
        if existing_result not in {"H", "D", "A"}:
            continue
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        if pd.isna(match_date):
            continue
        if match_date.date() <= today_date:
            continue
        frame.at[idx, "actual_home_goals"] = None
        frame.at[idx, "actual_away_goals"] = None
        frame.at[idx, "actual_result"] = None
        frame.at[idx, "is_correct"] = None
        frame.at[idx, "settled_at_utc"] = None
        cleared += 1
    return frame, cleared


def cleanup_settled_rows_not_in_results(frame, results_index):
    if frame is None or frame.empty:
        return frame, 0
    cleaned = 0
    for idx, row in frame.iterrows():
        existing_result = str(row.get("actual_result", "")).strip().upper()
        if existing_result not in {"H", "D", "A"}:
            continue

        date_key = str(row.get("match_date", "")).strip()
        comp_key = str(row.get("competition", "")).strip()
        home_key = normalize_team_key(row.get("home_team", ""))
        away_key = normalize_team_key(row.get("away_team", ""))
        if not date_key or not comp_key or not home_key or not away_key:
            continue

        key = (date_key, comp_key, home_key, away_key)
        key_rev = (date_key, comp_key, away_key, home_key)
        if key in results_index or key_rev in results_index:
            continue

        frame.at[idx, "actual_home_goals"] = None
        frame.at[idx, "actual_away_goals"] = None
        frame.at[idx, "actual_result"] = None
        frame.at[idx, "is_correct"] = None
        frame.at[idx, "settled_at_utc"] = None
        cleaned += 1
    return frame, cleaned


def cleanup_all_settled_rows(frame):
    if frame is None or frame.empty:
        return frame, 0
    cleaned = 0
    for idx, row in frame.iterrows():
        existing_result = str(row.get("actual_result", "")).strip().upper()
        if existing_result not in {"H", "D", "A"}:
            continue
        frame.at[idx, "actual_home_goals"] = None
        frame.at[idx, "actual_away_goals"] = None
        frame.at[idx, "actual_result"] = None
        frame.at[idx, "is_correct"] = None
        frame.at[idx, "settled_at_utc"] = None
        cleaned += 1
    return frame, cleaned


def resolve_espn_team_name(raw_name, competition, mapping_by_competition, predicted_team_names):
    raw = str(raw_name or "").strip()
    if not raw:
        return ""
    comp_map = mapping_by_competition.get(competition, {}) if isinstance(mapping_by_competition, dict) else {}

    direct = str(comp_map.get(raw, "")).strip()
    if direct:
        return direct, True

    raw_key = normalize_team_key(raw)

    # Try normalized lookup against mapping keys.
    for map_key, map_val in comp_map.items():
        if normalize_team_key(map_key) == raw_key:
            mapped = str(map_val).strip()
            if mapped:
                return mapped, True

    # Direct match against prediction team names.
    if raw in predicted_team_names:
        return raw, True

    by_key = {}
    for team in predicted_team_names:
        by_key.setdefault(normalize_team_key(team), []).append(team)
    candidates = by_key.get(raw_key, [])
    if len(candidates) == 1:
        return candidates[0], True

    close = difflib.get_close_matches(raw_key, list(by_key.keys()), n=1, cutoff=0.9)
    if close:
        teams = by_key.get(close[0], [])
        if len(teams) == 1:
            return teams[0], True

    return raw, False


def build_results_index_from_espn(predictions_df, mapping_by_competition):
    if predictions_df is None or predictions_df.empty:
        return {}, {}, {}
    results = {}
    mapping_updates = {}
    unresolved = {}
    for competition in sorted(set(predictions_df["competition"].astype(str).str.strip())):
        if not competition:
            continue
        league_key = ESPN_COMPETITION_KEYS.get(competition)
        if not league_key:
            print(f"Skipping {competition}: no ESPN league mapping configured.")
            continue
        unresolved.setdefault(competition, set())
        subset = predictions_df[predictions_df["competition"].astype(str).str.strip() == competition]
        if subset.empty:
            continue
        predicted_team_names = set(subset["home_team"].astype(str)) | set(subset["away_team"].astype(str))
        date_series = pd.to_datetime(subset["match_date"], errors="coerce")
        date_series = date_series[date_series.notna()]
        if date_series.empty:
            continue
        base_days = sorted(set(pd.Timestamp(dt).normalize() for dt in date_series))
        query_days = set(base_days)
        if competition == MLS_COMPETITION:
            # ESPN event timestamps can roll a fixture across UTC date boundaries.
            for day in base_days:
                query_days.add(day - pd.Timedelta(days=1))
                query_days.add(day + pd.Timedelta(days=1))
        date_codes = sorted(day.strftime("%Y%m%d") for day in query_days)
        for yyyymmdd in date_codes:
            url = f"{ESPN_BASE}/{league_key}/scoreboard?dates={yyyymmdd}"
            try:
                data = fetch_json(url, timeout=45)
            except Exception as error:
                print(f"Skipping {competition} date {yyyymmdd}: {error}")
                continue
            events = data.get("events", [])
            if not isinstance(events, list):
                continue

            for event in events:
                dt = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
                if pd.isna(dt):
                    continue
                date_key = event_date_key_for_competition(dt, competition)

                comps = event.get("competitions", [])
                if not comps:
                    continue
                comp0 = comps[0] or {}
                status_type = ((comp0.get("status") or {}).get("type") or {})
                if not bool(status_type.get("completed")):
                    continue

                competitors = comp0.get("competitors", [])
                home_name = ""
                away_name = ""
                home_score = None
                away_score = None
                for c in competitors:
                    side = str(c.get("homeAway", "")).strip().lower()
                    team_name = str((c.get("team") or {}).get("displayName") or "").strip()
                    score_val = pd.to_numeric(c.get("score"), errors="coerce")
                    if side == "home":
                        home_name, home_ok = resolve_espn_team_name(
                            team_name, competition, mapping_by_competition, predicted_team_names
                        )
                        if home_ok and home_name in predicted_team_names and team_name and team_name != home_name:
                            mapping_updates.setdefault(competition, {})
                            mapping_updates[competition].setdefault(team_name, home_name)
                        if not home_ok:
                            unresolved[competition].add(team_name)
                        home_score = int(score_val) if pd.notna(score_val) else None
                    elif side == "away":
                        away_name, away_ok = resolve_espn_team_name(
                            team_name, competition, mapping_by_competition, predicted_team_names
                        )
                        if away_ok and away_name in predicted_team_names and team_name and team_name != away_name:
                            mapping_updates.setdefault(competition, {})
                            mapping_updates[competition].setdefault(team_name, away_name)
                        if not away_ok:
                            unresolved[competition].add(team_name)
                        away_score = int(score_val) if pd.notna(score_val) else None

                if not home_name or not away_name or home_score is None or away_score is None:
                    continue

                key = (date_key, competition, normalize_team_key(home_name), normalize_team_key(away_name))
                results[key] = {
                    "actual_home_goals": home_score,
                    "actual_away_goals": away_score,
                    "actual_result": infer_result_code(home_score, away_score),
                    "completed": True,
                }
    unresolved = {k: sorted(v) for k, v in unresolved.items() if v}
    return results, mapping_updates, unresolved


def apply_mapping_updates(base_mapping, updates):
    merged = dict(base_mapping) if isinstance(base_mapping, dict) else {}
    added = 0
    changed = 0
    for competition, values in (updates or {}).items():
        merged.setdefault(competition, {})
        if not isinstance(merged[competition], dict):
            merged[competition] = {}
        for api_name, mapped_name in values.items():
            api = str(api_name).strip()
            mapped = str(mapped_name).strip()
            if not api or not mapped:
                continue
            existing = str(merged[competition].get(api, "")).strip()
            if not existing:
                merged[competition][api] = mapped
                added += 1
            elif existing != mapped:
                # Keep existing user mapping, but count drift.
                changed += 1
    return merged, added, changed


def main():
    args = parse_cli_args()

    global_df = load_predictions(GLOBAL_PREDICTIONS_FILE)
    mls_df = load_predictions(MLS_PREDICTIONS_FILE)

    if global_df is None and mls_df is None:
        raise ValueError("No predictions CSV files found to update.")

    global_mapping = load_mapping(GLOBAL_MAPPING_FILE)
    mls_mapping = load_mapping(MLS_MAPPING_FILE)

    needs_resettle = (not args.cleanup_all) or args.resettle_after_cleanup
    if needs_resettle:
        global_results, global_mapping_updates, global_unresolved = build_results_index_from_espn(global_df, global_mapping)
        mls_results, mls_mapping_updates, mls_unresolved = build_results_index_from_espn(mls_df, mls_mapping)
    else:
        global_results, global_mapping_updates, global_unresolved = {}, {}, {}
        mls_results, mls_mapping_updates, mls_unresolved = {}, {}, {}

    global_mapping, global_added, global_drift = apply_mapping_updates(global_mapping, global_mapping_updates)
    mls_mapping, mls_added, mls_drift = apply_mapping_updates(mls_mapping, mls_mapping_updates)
    save_mapping(GLOBAL_MAPPING_FILE, global_mapping)
    save_mapping(MLS_MAPPING_FILE, mls_mapping)

    global_updates = 0
    global_cleaned = 0
    if global_df is not None:
        if args.cleanup_all:
            global_df, global_cleaned = cleanup_all_settled_rows(global_df)
        if needs_resettle:
            global_df, global_updates = update_frame_with_results(global_df, global_results)
        global_df.to_csv(GLOBAL_PREDICTIONS_FILE, index=False)

    mls_updates = 0
    mls_cleaned = 0
    mls_future_cleared = 0
    if mls_df is not None:
        today_et = datetime.now(EASTERN_TZ).date()
        mls_df, mls_future_cleared = clear_future_settled_rows(mls_df, today_et)
        if args.cleanup_all:
            mls_df, mls_cleaned = cleanup_all_settled_rows(mls_df)
        elif args.cleanup_mls:
            mls_df, mls_cleaned = cleanup_settled_rows_not_in_results(mls_df, mls_results)
        if needs_resettle:
            mls_df, mls_updates = update_frame_with_results(mls_df, mls_results)
        mls_df.to_csv(MLS_PREDICTIONS_FILE, index=False)

    if args.cleanup_all:
        print(f"Global predictions cleanup cleared: {global_cleaned}")
        print(f"MLS predictions cleanup cleared: {mls_cleaned}")
    print(f"Global mapping auto-added: {global_added} (drift detected: {global_drift})")
    print(f"MLS mapping auto-added: {mls_added} (drift detected: {mls_drift})")
    if global_unresolved:
        print(f"Global unresolved ESPN names by league: {global_unresolved}")
    if mls_unresolved:
        print(f"MLS unresolved ESPN names by league: {mls_unresolved}")
    print(f"MLS future-settled rows cleared: {mls_future_cleared}")
    print(f"Global predictions updated: {global_updates}")
    if args.cleanup_mls and not args.cleanup_all:
        print(f"MLS predictions cleanup cleared: {mls_cleaned}")
    print(f"MLS predictions updated: {mls_updates}")
    print("Done.")


if __name__ == "__main__":
    main()
