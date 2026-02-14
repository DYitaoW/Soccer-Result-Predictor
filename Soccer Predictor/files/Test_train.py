import json
import os
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


PROCESSED_DIR = os.path.join("Data", "Processed_Data")
SEASON_PATTERN = re.compile(r"^premstat(\d{4})-(\d{2})\.csv$")
MIN_START_YEAR = 2002


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def per_game(stats, total_key, games_key):
    games = stats.get(games_key, 0)
    if not games:
        return 0.0
    return stats.get(total_key, 0) / games


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

# function used to setup the progrma
def build_features(match_df, season_key, overall_teams, season_teams, head_to_head):
    season_lookup = season_teams.get(season_key, {})
    rows = []
    previous_odds = {}

    for _, match in match_df.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        home_overall = overall_teams.get(home, {})
        away_overall = overall_teams.get(away, {})
        home_season = season_lookup.get(home, {})
        away_season = season_lookup.get(away, {})
        h2h_home = head_to_head.get(home, {}).get(away, {})

        home_prev = previous_odds.get(home, {"win": 0.0, "draw": 0.0, "lose": 0.0})
        away_prev = previous_odds.get(away, {"win": 0.0, "draw": 0.0, "lose": 0.0})

        avg_h = match.get("AvgH", 0.0)
        avg_d = match.get("AvgD", 0.0)
        avg_a = match.get("AvgA", 0.0)

        avg_h = 0.0 if pd.isna(avg_h) else float(avg_h)
        avg_d = 0.0 if pd.isna(avg_d) else float(avg_d)
        avg_a = 0.0 if pd.isna(avg_a) else float(avg_a)

        rows.append(
            {
                "home_overall_avg_goals": home_overall.get("avg_goals_scored", 0.0),
                "away_overall_avg_goals": away_overall.get("avg_goals_scored", 0.0),
                "home_overall_avg_conceded": home_overall.get("avg_goals_conceded", 0.0),
                "away_overall_avg_conceded": away_overall.get("avg_goals_conceded", 0.0),
                "home_home_goals_per_game": per_game(home_overall, "home_goals_scored", "home_games"),
                "away_away_goals_per_game": per_game(away_overall, "away_goals_scored", "away_games"),
                "home_season_points_per_game": per_game(home_season, "points", "games"),
                "away_season_points_per_game": per_game(away_season, "points", "games"),
                "h2h_home_win_rate": per_game(h2h_home, "wins", "games"),
                "h2h_goal_diff_per_game": per_game(h2h_home, "goals_scored", "games")
                - per_game(h2h_home, "goals_conceded", "games"),
                "home_prev_win_odds": home_prev["win"],
                "home_prev_draw_odds": home_prev["draw"],
                "home_prev_lose_odds": home_prev["lose"],
                "away_prev_win_odds": away_prev["win"],
                "away_prev_draw_odds": away_prev["draw"],
                "away_prev_lose_odds": away_prev["lose"],
            }
        )

        previous_odds[home] = {"win": avg_h, "draw": avg_d, "lose": avg_a}
        previous_odds[away] = {"win": avg_a, "draw": avg_d, "lose": avg_h}

    return pd.DataFrame(rows)

# functino used to ge thte values form the files
def load_matches(processed_dir):
    season_frames = []
    valid_files = []
    for name in os.listdir(processed_dir):
        start_year = parse_season_start_year(name)
        if start_year is not None:
            valid_files.append((start_year, name))
    valid_files.sort(key=lambda item: item[0])
    season_files = [name for _, name in valid_files]

    for file_name in season_files:
        file_path = os.path.join(processed_dir, file_name)
        frame = pd.read_csv(file_path)

        for odds_col in ["AvgH", "AvgD", "AvgA"]:
            if odds_col not in frame.columns:
                frame[odds_col] = 0.0

        frame = frame[["HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"]].dropna(
            subset=["HomeTeam", "AwayTeam", "FTR"]
        )

        frame["season_key"] = file_name.replace(".csv", "")
        season_frames.append(frame)

    if not season_frames:
        raise ValueError("No processed season CSV files were found.")

    return pd.concat(season_frames, ignore_index=True), season_files

# run and test the results
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, PROCESSED_DIR)
    overall_json = os.path.join(project_root, "Data", "Team_Data", "overall_teams.json")
    season_json = os.path.join(project_root, "Data", "Team_Data", "season_teams.json")
    h2h_json = os.path.join(project_root, "Data", "Team_Data", "head_to_head.json")

    overall_teams = load_json(overall_json)
    season_teams = load_json(season_json)
    head_to_head = load_json(h2h_json)

    matches, season_files = load_matches(processed_dir)
    latest_season_file = season_files[-1]
    latest_season_key = latest_season_file.replace(".csv", "")

    feature_frames = []
    for season_key, season_matches in matches.groupby("season_key", sort=False):
        feature_frames.append(
            build_features(
                season_matches[["HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"]],
                season_key,
                overall_teams,
                season_teams,
                head_to_head,
            )
        )

    X = pd.concat(feature_frames, ignore_index=True)
    y = matches["FTR"].reset_index(drop=True)

    train_mask = matches["season_key"] != latest_season_key
    test_mask = matches["season_key"] == latest_season_key

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    print("Train seasons:", len(season_files) - 1)
    print("Test season:", latest_season_file)
    print("Rows used:", len(X))
    print("Train rows:", len(X_train), "| Test rows:", len(X_test))
    print("Accuracy:", model.score(X_test, y_test))


if __name__ == "__main__":
    main()
