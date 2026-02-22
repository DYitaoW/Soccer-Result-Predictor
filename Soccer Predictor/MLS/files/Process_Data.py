import os
import re
from datetime import datetime

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(BASE_DIR, "Data", "Raw_Data")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "Data", "Processed_Data")
SEASON_PATTERN = re.compile(r"^(?:[a-z0-9]+stat)(\d{4})\.csv$", re.IGNORECASE)
MIN_START_YEAR = 2002
MIN_ROWS = 100
CURRENT_SEASON_MIN_ROWS = 20

REQUIRED_RAW_COLUMNS = {"Date", "Home", "Away", "HG", "AG", "Res"}

RESULT_MAP = {"H": 2, "D": 1, "A": 0}
ODDS_HOME_CANDIDATES = ["AvgCH", "PSCH", "B365CH", "MaxCH"]
ODDS_DRAW_CANDIDATES = ["AvgCD", "PSCD", "B365CD", "MaxCD"]
ODDS_AWAY_CANDIDATES = ["AvgCA", "PSCA", "B36CA", "B365CA", "MaxCA"]

OUTPUT_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "HST",
    "AS",
    "AST",
    "AvgH",
    "AvgD",
    "AvgA",
    "HomePointsBefore",
    "AwayPointsBefore",
    "HomeLeaguePosBefore",
    "AwayLeaguePosBefore",
    "ResultNum",
]


def parse_season_start_year(file_name):
    match = SEASON_PATTERN.match(file_name)
    if not match:
        return None

    start_year = int(match.group(1))
    if start_year < MIN_START_YEAR:
        return None
    if start_year > datetime.now().year:
        return None
    return start_year


def get_target_season_files(folder):
    valid = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if not file_name.endswith(".csv"):
                continue
            start_year = parse_season_start_year(file_name)
            if start_year is None:
                continue
            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path, folder)
            valid.append((start_year, rel_path))
    valid.sort(key=lambda item: item[0])
    return [name for _, name in valid]


def first_existing_value(row, candidates, default=0.0):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                continue
    return float(default)


def has_minimum_rows(df, season_start_year):
    current_year = datetime.now().year
    in_progress = season_start_year == (current_year - 1)
    minimum = CURRENT_SEASON_MIN_ROWS if in_progress else MIN_ROWS
    return len(df) >= minimum


def add_table_context_columns(df):
    required = {"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        return df

    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    table = {team: {"points": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams}
    position_map = {team: idx + 1 for idx, team in enumerate(teams)}

    home_points_before = []
    away_points_before = []
    home_pos_before = []
    away_pos_before = []

    def rank_positions():
        ranked = sorted(
            teams,
            key=lambda t: (-table[t]["points"], -table[t]["gd"], -table[t]["gf"], t),
        )
        return {team: idx + 1 for idx, team in enumerate(ranked)}

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_points_before.append(float(table.get(home, {}).get("points", 0.0)))
        away_points_before.append(float(table.get(away, {}).get("points", 0.0)))
        home_pos_before.append(float(position_map.get(home, len(teams))))
        away_pos_before.append(float(position_map.get(away, len(teams))))

        hg = row.get("FTHG")
        ag = row.get("FTAG")
        ftr = row.get("FTR")
        if pd.isna(hg) or pd.isna(ag) or pd.isna(ftr):
            continue

        hg = int(hg)
        ag = int(ag)
        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg
        table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
        table[away]["gd"] = table[away]["gf"] - table[away]["ga"]

        if ftr == "H":
            table[home]["points"] += 3
        elif ftr == "A":
            table[away]["points"] += 3
        elif ftr == "D":
            table[home]["points"] += 1
            table[away]["points"] += 1

        position_map = rank_positions()

    df["HomePointsBefore"] = home_points_before
    df["AwayPointsBefore"] = away_points_before
    df["HomeLeaguePosBefore"] = home_pos_before
    df["AwayLeaguePosBefore"] = away_pos_before
    return df


def convert_mls_raw_to_standard(df):
    if not REQUIRED_RAW_COLUMNS.issubset(df.columns):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], dayfirst=False, format="mixed", errors="coerce")
    out["HomeTeam"] = df["Home"].astype(str).str.strip()
    out["AwayTeam"] = df["Away"].astype(str).str.strip()
    out["FTHG"] = pd.to_numeric(df["HG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(df["AG"], errors="coerce")
    out["FTR"] = df["Res"].astype(str).str.strip()

    # MLS source does not include shots in this feed; keep sentinel values.
    out["HS"] = "NA"
    out["HST"] = "NA"
    out["AS"] = "NA"
    out["AST"] = "NA"

    out["AvgH"] = [first_existing_value(row, ODDS_HOME_CANDIDATES, 0.0) for _, row in df.iterrows()]
    out["AvgD"] = [first_existing_value(row, ODDS_DRAW_CANDIDATES, 0.0) for _, row in df.iterrows()]
    out["AvgA"] = [first_existing_value(row, ODDS_AWAY_CANDIDATES, 0.0) for _, row in df.iterrows()]

    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    out = out[out["FTR"].isin({"H", "D", "A"})]
    if out.empty:
        return out

    out = out.sort_values("Date").reset_index(drop=True)
    out = add_table_context_columns(out)
    out["ResultNum"] = out["FTR"].map(RESULT_MAP)

    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out[OUTPUT_COLUMNS]


def main():
    if not os.path.isdir(RAW_FOLDER):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_FOLDER}")

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    target_files = get_target_season_files(RAW_FOLDER)
    if not target_files:
        raise ValueError("No valid MLS season files were found in Raw_Data.")

    for rel_path in target_files:
        file_path = os.path.join(RAW_FOLDER, rel_path)
        season_start_year = parse_season_start_year(os.path.basename(rel_path))
        if season_start_year is None:
            continue

        print(f"Processing {rel_path}...")
        try:
            raw_df = pd.read_csv(file_path)
        except Exception:
            raw_df = pd.read_csv(file_path, encoding="latin-1", engine="python", on_bad_lines="skip")

        if not has_minimum_rows(raw_df, season_start_year):
            continue

        processed_df = convert_mls_raw_to_standard(raw_df)
        if processed_df.empty:
            continue

        output_path = os.path.join(PROCESSED_FOLDER, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)

    print("All MLS files processed.")


if __name__ == "__main__":
    main()
