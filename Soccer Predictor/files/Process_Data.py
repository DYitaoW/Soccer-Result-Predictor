import pandas as pd
import os

raw_folder = "Data/Raw_Data"
processed_folder = "Data/Processed_Data"

columns = [
    "Date", "HomeTeam", "FTHG", "HTHG", "HS", "HST", "HC", "HF","HFKC", "HO", "HY", "HR", "HBP", # home team data
    "AwayTeam", "FTAG", "HTAG", "AS", "AST", "AC", "AF", "AFKC", "AO", "AY", "AR", "ABP", # away team data
    "Referee", "FTR", "HTR", # overall game data
    "AvgH", "AvgD", "AvgA", # overall betting odds data
    "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5", # goal over/under betting odds
    "AvgAHH", "AvgAHA" # overall handicap betting odds 
]

result_map = {"H": 2, "D": 1, "A": 0}

# loop through all csv files
for file in os.listdir(raw_folder):

    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(raw_folder, file)

    print(f"Processing {file}...")

    df = pd.read_csv(file_path)

    # convert date to all the same format
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # keep only available columns in case the data is missing
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]

    # sort the column by the date
    if "Date" in df.columns:
        df = df.sort_values("Date")

    # convert result to numeric numbers
    if "FTR" in df.columns:
        df["ResultNum"] = df["FTR"].map(result_map)

    # save processed file in the Processed_Data folder
    output_path = os.path.join(processed_folder, file)
    df.to_csv(output_path, index=False)

print("All files processed.")
