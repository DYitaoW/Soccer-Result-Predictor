import pandas as pd
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(BASE_DIR, "Data", "Raw_Data")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "Data", "Processed_Data")
SEASON_PATTERN = re.compile(r"^(?:[a-z0-9]+stat)(\d{4})-(\d{2})\.csv$", re.IGNORECASE)
GENERAL_REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "HST", "AS", "AST"]
MIN_COMPLETENESS_RATIO = 0.95
MIN_ROWS = 250
CURRENT_SEASON_MIN_ROWS = 20
MIN_START_YEAR = 2002

columns = [
    "Date", "HomeTeam", "FTHG", "HTHG", "HS", "HST", "HC", "HF","HFKC", "HO", "HY", "HR", "HBP", # home team data
    "AwayTeam", "FTAG", "HTAG", "AS", "AST", "AC", "AF", "AFKC", "AO", "AY", "AR", "ABP", # away team data
    "Referee", "FTR", "HTR", # overall game data
    "AvgH", "AvgD", "AvgA", # overall betting odds data
    "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5", # goal over/under betting odds
    "AvgAHH", "AvgAHA" # overall handicap betting odds 
]

result_map = {"H": 2, "D": 1, "A": 0}


def parse_season_start_year(file_name):
    match = SEASON_PATTERN.match(file_name)
    if not match:
        return None

    start_year = int(match.group(1))
    end_year_two_digits = int(match.group(2))
    if end_year_two_digits != (start_year + 1) % 100:
        return None
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
            if start_year is not None:
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, folder)
                valid.append((start_year, rel_path))

    valid.sort(key=lambda item: item[0])
    return [name for _, name in valid]


def has_required_general_data(df, start_year):
    if any(col not in df.columns for col in GENERAL_REQUIRED_COLUMNS):
        return False

    current_year = datetime.now().year
    in_progress_season = start_year == (current_year - 1)
    if in_progress_season:
        return len(df) >= CURRENT_SEASON_MIN_ROWS

    if len(df) < MIN_ROWS:
        return False

    complete_rows = df[GENERAL_REQUIRED_COLUMNS].notna().all(axis=1).mean()
    return complete_rows >= MIN_COMPLETENESS_RATIO


def add_table_context_columns(df):
    required = {"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        return df

    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    table = {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0}
        for team in teams
    }
    position_map = {team: idx + 1 for idx, team in enumerate(teams)}

    home_points_before = []
    away_points_before = []
    home_pos_before = []
    away_pos_before = []

    def rank_positions():
        ranked = sorted(
            teams,
            key=lambda t: (
                -table[t]["points"],
                -table[t]["gd"],
                -table[t]["gf"],
                t,
            ),
        )
        return {team: pos + 1 for pos, team in enumerate(ranked)}

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_points_before.append(float(table.get(home, {}).get("points", 0)))
        away_points_before.append(float(table.get(away, {}).get("points", 0)))
        home_pos_before.append(float(position_map.get(home, len(teams))))
        away_pos_before.append(float(position_map.get(away, len(teams))))

        if home not in table or away not in table:
            continue

        hg = row.get("FTHG")
        ag = row.get("FTAG")
        ftr = row.get("FTR")
        if pd.isna(hg) or pd.isna(ag) or pd.isna(ftr):
            continue

        hg = int(hg)
        ag = int(ag)
        table[home]["played"] += 1
        table[away]["played"] += 1
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

def main():
    if not os.path.isdir(RAW_FOLDER):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_FOLDER}")

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    target_files = get_target_season_files(RAW_FOLDER)
    if not target_files:
        raise ValueError("No valid files were found in Raw_Data.")

    # loop through selected csv files
    for rel_path in target_files:
        file_path = os.path.join(RAW_FOLDER, rel_path)
        print(f"Processing {rel_path}...")
        season_start_year = parse_season_start_year(os.path.basename(rel_path))
        if season_start_year is None:
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception:
            # Older files can have missing datapoints
            df = pd.read_csv(
                file_path,
                encoding="latin-1",
                engine="python",
                on_bad_lines="skip",
            )

        if not has_required_general_data(df, season_start_year):
            continue

        # convert the date to all the same format 
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")

        # keep only available columns in case the data is missing from a season
        available_columns = [col for col in columns if col in df.columns]
        df = df[available_columns]

        # sort the rows by date
        if "Date" in df.columns:
            df = df.sort_values("Date")

        # add standings context for each match before kickoff
        df = add_table_context_columns(df)

        # convert result to numeric numbers
        if "FTR" in df.columns:
            df["ResultNum"] = df["FTR"].map(result_map)

        # save processed file in the Processed_Data folder
        output_path = os.path.join(PROCESSED_FOLDER, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    print("All files processed.")


if __name__ == "__main__":
    main()
