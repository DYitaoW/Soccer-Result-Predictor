import pandas as pd
import os
import json
import re
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "Processed_Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "Team_Data")
SEASON_PATTERN = re.compile(r"^premstat(\d{4})-(\d{2})\.csv$")
MIN_START_YEAR = 2002

# the basic storage for all of the stats to be stored
def blank_team_stats():
    return {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,

        "goals_scored": 0,
        "goals_conceded": 0,

        "home_games": 0,
        "away_games": 0,

        "home_wins": 0,
        "home_draws": 0,
        "home_losses": 0,

        "away_wins": 0,
        "away_draws": 0,
        "away_losses": 0,

        "home_goals_scored": 0,
        "home_goals_conceded": 0,
        "away_goals_scored": 0,
        "away_goals_conceded": 0,

        "shots_for": 0,
        "shots_against": 0,
        "shots_on_target_for": 0,
        "shots_on_target_against": 0,

        "home_shots_for": 0,
        "home_shots_against": 0,
        "home_sot_for": 0,
        "home_sot_against": 0,

        "away_shots_for": 0,
        "away_shots_against": 0,
        "away_sot_for": 0,
        "away_sot_against": 0
    }

# the function used to update a team with the stats in the processed file
def update_team(stats, gf, ga, shots_f, shots_a, sot_f, sot_a, result, is_home):
    stats["games"] += 1
    stats["goals_scored"] += gf
    stats["goals_conceded"] += ga
    stats["shots_for"] += shots_f
    stats["shots_against"] += shots_a
    stats["shots_on_target_for"] += sot_f
    stats["shots_on_target_against"] += sot_a

    if result == "W":
        stats["wins"] += 1
        stats["points"] += 3
    elif result == "D":
        stats["draws"] += 1
        stats["points"] += 1
    else:
        stats["losses"] += 1

    if is_home:
        stats["home_games"] += 1
        stats["home_goals_scored"] += gf
        stats["home_goals_conceded"] += ga
        stats["home_shots_for"] += shots_f
        stats["home_shots_against"] += shots_a
        stats["home_sot_for"] += sot_f
        stats["home_sot_against"] += sot_a

        if result == "W":
            stats["home_wins"] += 1
        elif result == "D":
            stats["home_draws"] += 1
        else:
            stats["home_losses"] += 1
    else:
        stats["away_games"] += 1
        stats["away_goals_scored"] += gf
        stats["away_goals_conceded"] += ga
        stats["away_shots_for"] += shots_f
        stats["away_shots_against"] += shots_a
        stats["away_sot_for"] += sot_f
        stats["away_sot_against"] += sot_a

        if result == "W":
            stats["away_wins"] += 1
        elif result == "D":
            stats["away_draws"] += 1
        else:
            stats["away_losses"] += 1


# function used to calculate the average goals for/agains for a team
def calculate_averages(team_dict):
    for team, stats in team_dict.items():
        g = stats["games"]
        if g > 0:
            stats["avg_goals_scored"] = round(stats["goals_scored"] / g, 2)
            stats["avg_goals_conceded"] = round(stats["goals_conceded"] / g, 2)
            stats["avg_shots"] = round(stats["shots_for"] / g, 2)
            stats["avg_sot"] = round(stats["shots_on_target_for"] / g, 2)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)

# gets the year for the file
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

def get_target_season_files():
    valid = []
    for file_name in os.listdir(PROCESSED_DIR):
        start_year = parse_season_start_year(file_name)
        if start_year is not None:
            valid.append((start_year, file_name))
    valid.sort(key=lambda item: item[0])
    return [name for _, name in valid]

# function used to build a file to store the current (last 5 game) stats for each team in the current season
def build_current_form_file():
    files = get_target_season_files()
    if not files:
        raise ValueError("No processed season CSV files found.")

    latest_file = files[-1]
    latest_path = os.path.join(PROCESSED_DIR, latest_file)
    df = pd.read_csv(latest_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

    team_matches = defaultdict(list)

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        hg = row["FTHG"]
        ag = row["FTAG"]
        result = row["FTR"]

        avg_h = row.get("AvgH")
        avg_d = row.get("AvgD")
        avg_a = row.get("AvgA")

        if result == "H":
            home_res = "W"
            away_res = "L"
        elif result == "A":
            home_res = "L"
            away_res = "W"
        else:
            home_res = "D"
            away_res = "D"

        team_matches[home].append(
            {
                "result": home_res,
                "gf": hg,
                "ga": ag,
                "win_odds": safe_float(avg_h),
                "draw_odds": safe_float(avg_d),
                "lose_odds": safe_float(avg_a),
            }
        )
        team_matches[away].append(
            {
                "result": away_res,
                "gf": ag,
                "ga": hg,
                "win_odds": safe_float(avg_a),
                "draw_odds": safe_float(avg_d),
                "lose_odds": safe_float(avg_h),
            }
        )

    current_form = {"season": latest_file.replace(".csv", ""), "teams": {}}

    for team, matches in team_matches.items():
        recent = matches[-5:]

        wins = sum(1 for match in recent if match["result"] == "W")
        draws = sum(1 for match in recent if match["result"] == "D")
        losses = sum(1 for match in recent if match["result"] == "L")
        points = wins * 3 + draws

        goals_for = sum(match["gf"] for match in recent)
        goals_against = sum(match["ga"] for match in recent)
        recent_count = len(recent)

        last_game = matches[-1] if matches else {}

        current_form["teams"][team] = {
            "games_played": len(matches),
            "form_last_5": "".join(match["result"] for match in recent),
            "wins_last_5": wins,
            "draws_last_5": draws,
            "losses_last_5": losses,
            "points_last_5": points,
            "avg_goals_for_last_5": round(goals_for / recent_count, 2) if recent_count else 0.0,
            "avg_goals_against_last_5": round(goals_against / recent_count, 2) if recent_count else 0.0,
            "previous_match_win_odds": last_game.get("win_odds"),
            "previous_match_draw_odds": last_game.get("draw_odds"),
            "previous_match_lose_odds": last_game.get("lose_odds"),
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "current_form.json"), "w") as file:
        json.dump(current_form, file, indent=4)

    print(f"Current form written from {latest_file}")

# function used to sort the data and store for each season and team
# thsi function is the main one called that completes the entiree task
def sort_all_seasons():
    overall_teams = defaultdict(blank_team_stats)
    season_data = {}
    h2h_stats = defaultdict(lambda: defaultdict(blank_team_stats))

    files = get_target_season_files()

    for file in files:
        if not file.endswith(".csv"):
            continue

        print(f"Processing {file}")
        path = os.path.join(PROCESSED_DIR, file)
        df = pd.read_csv(path)

        season_teams = defaultdict(blank_team_stats)

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            hg = row["FTHG"]
            ag = row["FTAG"]

            hs = row["HS"]
            ass = row["AS"]

            hst = row["HST"]
            ast = row["AST"]

            result = row["FTR"]

            if result == "H":
                home_res = "W"
                away_res = "L"
            elif result == "A":
                home_res = "L"
                away_res = "W"
            else:
                home_res = away_res = "D"

            # update the stats for the teams overall, season, and head to head stats
            update_team(season_teams[home], hg, ag, hs, ass, hst, ast, home_res, True)
            update_team(season_teams[away], ag, hg, ass, hs, ast, hst, away_res, False)

            update_team(overall_teams[home], hg, ag, hs, ass, hst, ast, home_res, True)
            update_team(overall_teams[away], ag, hg, ass, hs, ast, hst, away_res, False)

            update_team(h2h_stats[home][away], hg, ag, hs, ass, hst, ast, home_res, True)
            update_team(h2h_stats[away][home], ag, hg, ass, hs, ast, hst, away_res, False)

        calculate_averages(season_teams)
        season_data[file.replace(".csv", "")] = season_teams

    calculate_averages(overall_teams)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "overall_teams.json"), "w") as f:
        json.dump(overall_teams, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "season_teams.json"), "w") as f:
        json.dump(season_data, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "head_to_head.json"), "w") as f:
        json.dump(h2h_stats, f, indent=4)

    print("\nDone!\n")


# testing
if __name__ == "__main__":
    sort_all_seasons()
    build_current_form_file()
