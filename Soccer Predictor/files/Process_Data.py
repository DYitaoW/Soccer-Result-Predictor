import pandas as pd
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(BASE_DIR, "Data", "Raw_Data")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "Data", "Processed_Data")
SEASON_PATTERN = re.compile(r"^premstat(\d{4})-(\d{2})\.csv$")
GENERAL_REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "HST", "AS", "AST"]
MIN_COMPLETENESS_RATIO = 0.95
MIN_ROWS = 250
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
    for file_name in os.listdir(folder):
        start_year = parse_season_start_year(file_name)
        if start_year is not None:
            valid.append((start_year, file_name))

    valid.sort(key=lambda item: item[0])
    return [name for _, name in valid]


def has_required_general_data(df):
    if len(df) < MIN_ROWS:
        return False
    if any(col not in df.columns for col in GENERAL_REQUIRED_COLUMNS):
        return False

    complete_rows = df[GENERAL_REQUIRED_COLUMNS].notna().all(axis=1).mean()
    return complete_rows >= MIN_COMPLETENESS_RATIO

def main():
    if not os.path.isdir(RAW_FOLDER):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_FOLDER}")

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    target_files = get_target_season_files(RAW_FOLDER)
    if not target_files:
        raise ValueError("No valid files were found in Raw_Data.")

    # loop through selected csv files
    for file in target_files:

        file_path = os.path.join(RAW_FOLDER, file)
        print(f"Processing {file}...")

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

        if not has_required_general_data(df):
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

        # convert result to numeric numbers
        if "FTR" in df.columns:
            df["ResultNum"] = df["FTR"].map(result_map)

        # save processed file in the Processed_Data folder
        output_path = os.path.join(PROCESSED_FOLDER, file)
        df.to_csv(output_path, index=False)

    print("All files processed.")


if __name__ == "__main__":
    main()
