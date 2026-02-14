import os
import re
import urllib.request
import pandas as pd
from io import StringIO


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Data", "Raw_Data")
DATA_PAGE_URL = "https://www.football-data.co.uk/englandm.php"
GENERAL_REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "HST", "AS", "AST"]
MIN_COMPLETENESS_RATIO = 0.95
MIN_ROWS = 250
MIN_START_YEAR = 2002


def fetch_text(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def season_code_to_label(code):
    start_two_digits = int(code[:2])
    end_two_digits = int(code[2:])

    if start_two_digits >= 90:
        start = 1900 + start_two_digits
    else:
        start = 2000 + start_two_digits

    if end_two_digits >= 90:
        end = 1900 + end_two_digits
    else:
        end = 2000 + end_two_digits

    return f"{start}-{str(end)[-2:]}"


def season_code_to_start_year(code):
    start_two_digits = int(code[:2])
    if start_two_digits >= 90:
        return 1900 + start_two_digits
    return 2000 + start_two_digits


def extract_league_links(page_html, league_code="E0"):
    pattern = re.compile(r'href="(mmz4281/(\d{4})/' + re.escape(league_code) + r'\.csv)"', re.IGNORECASE)
    links = {}

    for rel_path, season_code in pattern.findall(page_html):
        full_url = f"https://www.football-data.co.uk/{rel_path}"
        links[season_code] = full_url

    return links


def download_file(url, out_path):
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    with open(out_path, "wb") as file:
        file.write(content)
    return content


def has_required_general_data(csv_bytes):
    try:
        text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = csv_bytes.decode("latin-1", errors="replace")

    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        try:
            df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")
        except Exception:
            return False

    if len(df) < MIN_ROWS:
        return False
    if any(col not in df.columns for col in GENERAL_REQUIRED_COLUMNS):
        return False

    complete_rows = df[GENERAL_REQUIRED_COLUMNS].notna().all(axis=1).mean()
    return complete_rows >= MIN_COMPLETENESS_RATIO


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"Fetching {DATA_PAGE_URL} ...")
    page_html = fetch_text(DATA_PAGE_URL)
    links = extract_league_links(page_html, league_code="E0")

    if not links:
        raise RuntimeError("No Premier League CSV links were found on football-data.co.uk/data.php")

    seasons = sorted(links.keys())
    print(f"Found {len(seasons)} Premier League season links.")

    downloaded = 0
    kept_files = set()
    for season_code in seasons:
        start_year = season_code_to_start_year(season_code)
        if start_year < MIN_START_YEAR:
            continue

        url = links[season_code]
        season_label = season_code_to_label(season_code)
        out_name = f"premstat{season_label}.csv"
        out_path = os.path.join(RAW_DATA_DIR, out_name)

        print(f"Checking {season_label} -> {out_name}")
        csv_bytes = download_file(url, out_path)
        if has_required_general_data(csv_bytes):
            kept_files.add(out_name)
            downloaded += 1
            continue

        print(f"Skipping {out_name}: core stats are incomplete.")
        os.remove(out_path)

    # Remove stale CSV files that are not valid from the latest check.
    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".csv") and file_name not in kept_files:
            os.remove(os.path.join(RAW_DATA_DIR, file_name))

    print(f"\nDone. Kept {downloaded} valid file(s) in {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
