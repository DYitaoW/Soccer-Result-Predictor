import json
import os
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "Processed_Data")
TEAM_DATA_DIR = os.path.join(BASE_DIR, "Data", "Team_Data")
SEASON_PATTERN = re.compile(r"^premstat(\d{4})-(\d{2})\.csv$")
MIN_START_YEAR = 2002 # start year for english teams to stop downloading the unusable files


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        return load_json(path)
    except Exception:
        return None


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


def build_fallback_data(matches, season_files):
    overall_teams = defaultdict(
        lambda: {
            "games": 0,
            "goals_scored": 0,
            "goals_conceded": 0,
            "home_games": 0,
            "away_games": 0,
            "home_goals_scored": 0,
            "away_goals_scored": 0,
        }
    )
    season_teams = defaultdict(lambda: defaultdict(lambda: {"games": 0, "points": 0}))
    head_to_head = defaultdict(
        lambda: defaultdict(lambda: {"games": 0, "wins": 0, "goals_scored": 0, "goals_conceded": 0})
    )

    for _, row in matches.iterrows():
        season_key = row["season_key"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg = float(row["FTHG"])
        ag = float(row["FTAG"])
        result = row["FTR"]

        overall_teams[home]["games"] += 1
        overall_teams[home]["goals_scored"] += hg
        overall_teams[home]["goals_conceded"] += ag
        overall_teams[home]["home_games"] += 1
        overall_teams[home]["home_goals_scored"] += hg

        overall_teams[away]["games"] += 1
        overall_teams[away]["goals_scored"] += ag
        overall_teams[away]["goals_conceded"] += hg
        overall_teams[away]["away_games"] += 1
        overall_teams[away]["away_goals_scored"] += ag

        season_teams[season_key][home]["games"] += 1
        season_teams[season_key][away]["games"] += 1

        head_to_head[home][away]["games"] += 1
        head_to_head[home][away]["goals_scored"] += hg
        head_to_head[home][away]["goals_conceded"] += ag
        head_to_head[away][home]["games"] += 1
        head_to_head[away][home]["goals_scored"] += ag
        head_to_head[away][home]["goals_conceded"] += hg

        if result == "H":
            season_teams[season_key][home]["points"] += 3
            head_to_head[home][away]["wins"] += 1
        elif result == "A":
            season_teams[season_key][away]["points"] += 3
            head_to_head[away][home]["wins"] += 1
        else:
            season_teams[season_key][home]["points"] += 1
            season_teams[season_key][away]["points"] += 1

    for team, stats in overall_teams.items():
        games = max(stats["games"], 1)
        stats["avg_goals_scored"] = stats["goals_scored"] / games
        stats["avg_goals_conceded"] = stats["goals_conceded"] / games

    current_form = {"season": season_files[-1].replace(".csv", ""), "teams": {}}
    latest_key = current_form["season"]
    latest_matches = matches[matches["season_key"] == latest_key]
    last_odds = {}
    for _, row in latest_matches.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        last_odds[home] = {
            "previous_match_win_odds": float(row.get("AvgH", 0.0) or 0.0),
            "previous_match_draw_odds": float(row.get("AvgD", 0.0) or 0.0),
            "previous_match_lose_odds": float(row.get("AvgA", 0.0) or 0.0),
        }
        last_odds[away] = {
            "previous_match_win_odds": float(row.get("AvgA", 0.0) or 0.0),
            "previous_match_draw_odds": float(row.get("AvgD", 0.0) or 0.0),
            "previous_match_lose_odds": float(row.get("AvgH", 0.0) or 0.0),
        }
    current_form["teams"] = last_odds

    return dict(overall_teams), {k: dict(v) for k, v in season_teams.items()}, {
        k: dict(v) for k, v in head_to_head.items()
    }, current_form


def build_features(match_df, season_key, overall_teams, season_teams, head_to_head, current_form):
    season_lookup = season_teams.get(season_key, {})
    form_lookup = current_form.get("teams", {})
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

        home_prev = previous_odds.get(home)
        away_prev = previous_odds.get(away)

        if home_prev is None:
            form_home = form_lookup.get(home, {})
            home_prev = {
                "win": float(form_home.get("previous_match_win_odds") or 0.0),
                "draw": float(form_home.get("previous_match_draw_odds") or 0.0),
                "lose": float(form_home.get("previous_match_lose_odds") or 0.0),
            }

        if away_prev is None:
            form_away = form_lookup.get(away, {})
            away_prev = {
                "win": float(form_away.get("previous_match_win_odds") or 0.0),
                "draw": float(form_away.get("previous_match_draw_odds") or 0.0),
                "lose": float(form_away.get("previous_match_lose_odds") or 0.0),
            }

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


def load_training_matches(processed_dir):
    frames = []
    valid_files = []
    for name in os.listdir(processed_dir):
        start_year = parse_season_start_year(name)
        if start_year is not None:
            valid_files.append((start_year, name))
    valid_files.sort(key=lambda item: item[0])
    season_files = [name for _, name in valid_files]

    for file_name in season_files:
        path = os.path.join(processed_dir, file_name)
        df = pd.read_csv(path)

        for col in ["AvgH", "AvgD", "AvgA"]:
            if col not in df.columns:
                df[col] = 0.0

        df = df[
            ["HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "AvgH", "AvgD", "AvgA"]
        ].dropna(subset=["HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST"])
        df["season_key"] = file_name.replace(".csv", "")
        frames.append(df)

    if not frames:
        raise ValueError("No season CSV files found in Data/Processed_Data.")

    return pd.concat(frames, ignore_index=True), season_files


def resolve_team_name(raw_name, valid_names):
    if not raw_name:
        return None

    def normalize(name):
        name = name.lower().strip()
        name = name.replace("&", "and")
        name = re.sub(r"[^a-z0-9]+", "", name)
        return name

    alias_map = {}
    for team in valid_names:
        alias_map[normalize(team)] = team

    manual_aliases = {
        "manutd": "Man United",
        "manchesterunited": "Man United",
        "mancity": "Man City",
        "manchestercity": "Man City",
        "spurs": "Tottenham",
        "tottenhamhotspur": "Tottenham",
        "wolves": "Wolves",
        "wolverhampton": "Wolves",
        "wolverhamptonwanderers": "Wolves",
        "newcastleutd": "Newcastle",
        "newcastleunited": "Newcastle",
        "nottinghamforest": "Nott'm Forest",
        "nottmforest": "Nott'm Forest",
        "westbrom": "West Brom",
        "westbromwich": "West Brom",
        "westbromwichalbion": "West Brom",
        "sheffieldunited": "Sheffield United",
        "sheffieldutd": "Sheffield United",
        "sheffieldwednesday": "Sheffield Weds",
        "sheffwednesday": "Sheffield Weds",
        "sheffwed": "Sheffield Weds",
        "qpr": "QPR",
        "leicestercity": "Leicester",
        "leedsunited": "Leeds",
        "ipswichtown": "Ipswich",
        "lutontown": "Luton",
        "brightonandhovealbion": "Brighton",
    }

    for alias_key, canonical in manual_aliases.items():
        if canonical in valid_names:
            alias_map[alias_key] = canonical

    key = normalize(raw_name)
    direct = alias_map.get(key)
    if direct:
        return direct

    # Loose fallback: contains match on normalized names.
    candidates = [team for team in valid_names if key and key in normalize(team)]
    if len(candidates) == 1:
        return candidates[0]

    return None


def build_match_input(home_team, away_team):
    return pd.DataFrame(
        [{"HomeTeam": home_team, "AwayTeam": away_team, "FTR": "D", "AvgH": 0.0, "AvgD": 0.0, "AvgA": 0.0}]
    )


def main():
    matches, season_files = load_training_matches(PROCESSED_DIR)
    if not season_files:
        raise ValueError("No valid processed season files were found for prediction.")

    overall_teams = load_json_if_exists(os.path.join(TEAM_DATA_DIR, "overall_teams.json"))
    season_teams = load_json_if_exists(os.path.join(TEAM_DATA_DIR, "season_teams.json"))
    head_to_head = load_json_if_exists(os.path.join(TEAM_DATA_DIR, "head_to_head.json"))
    current_form = load_json_if_exists(os.path.join(TEAM_DATA_DIR, "current_form.json"))

    if (
        overall_teams is None
        or season_teams is None
        or head_to_head is None
        or current_form is None
        or not isinstance(overall_teams, dict)
        or len(overall_teams) == 0
    ):
        (
            fallback_overall,
            fallback_season,
            fallback_h2h,
            fallback_form,
        ) = build_fallback_data(matches, season_files)
        overall_teams = fallback_overall if overall_teams is None else overall_teams
        season_teams = fallback_season if season_teams is None else season_teams
        head_to_head = fallback_h2h if head_to_head is None else head_to_head
        current_form = fallback_form if current_form is None else current_form

    feature_frames = []
    for season_key, season_matches in matches.groupby("season_key", sort=False):
        feature_frames.append(
            build_features(
                season_matches[["HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"]],
                season_key,
                overall_teams,
                season_teams,
                head_to_head,
                current_form,
            )
        )

    X = pd.concat(feature_frames, ignore_index=True)
    y_result = matches["FTR"].reset_index(drop=True)
    y_home_goals = matches["FTHG"].reset_index(drop=True)
    y_away_goals = matches["FTAG"].reset_index(drop=True)
    y_home_shots = matches["HS"].reset_index(drop=True)
    y_away_shots = matches["AS"].reset_index(drop=True)
    y_home_sot = matches["HST"].reset_index(drop=True)
    y_away_sot = matches["AST"].reset_index(drop=True)

    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    home_goal_reg = RandomForestRegressor(n_estimators=400, random_state=42)
    away_goal_reg = RandomForestRegressor(n_estimators=400, random_state=43)
    home_shot_reg = RandomForestRegressor(n_estimators=350, random_state=44)
    away_shot_reg = RandomForestRegressor(n_estimators=350, random_state=45)
    home_sot_reg = RandomForestRegressor(n_estimators=350, random_state=46)
    away_sot_reg = RandomForestRegressor(n_estimators=350, random_state=47)

    clf.fit(X, y_result)
    home_goal_reg.fit(X, y_home_goals)
    away_goal_reg.fit(X, y_away_goals)
    home_shot_reg.fit(X, y_home_shots)
    away_shot_reg.fit(X, y_away_shots)
    home_sot_reg.fit(X, y_home_sot)
    away_sot_reg.fit(X, y_away_sot)

    available_teams = sorted(overall_teams.keys())
    if not available_teams:
        available_teams = sorted(
            set(matches["HomeTeam"].dropna().tolist()) | set(matches["AwayTeam"].dropna().tolist())
        )

    print("\nMatch Predictor\n")
    print("Enter teams for prediction. Team 1 is always the home team.")
    print("Type 'q' to quit.\n")

    latest_season = season_files[-1].replace(".csv", "")

    while True:
        team_1_raw = input("Team 1 (Home): ").strip()
        if team_1_raw.lower() in {"q"}:
            print("\nExiting predictor.")
            break

        team_2_raw = input("Team 2 (Away): ").strip()
        if team_2_raw.lower() in {"q"}:
            print("\nExiting predictor.")
            break

        team_1 = resolve_team_name(team_1_raw, available_teams)
        team_2 = resolve_team_name(team_2_raw, available_teams)

        if not team_1 or not team_2:
            print("\nOne or both team names were not recognized.")
            continue
        if team_1 == team_2:
            print("\nTeams must be different.\n")
            continue

        home_team, away_team = team_1, team_2
        match_input = build_match_input(home_team, away_team)
        X_match = build_features(
            match_input,
            latest_season,
            overall_teams,
            season_teams,
            head_to_head,
            current_form,
        )

        probabilities = {"H": 0.0, "D": 0.0, "A": 0.0}
        proba_values = clf.predict_proba(X_match)[0]
        for idx, label in enumerate(clf.classes_):
            probabilities[label] = float(proba_values[idx])

        prediction = clf.predict(X_match)[0]
        predicted_home_goals = max(0.0, float(home_goal_reg.predict(X_match)[0]))
        predicted_away_goals = max(0.0, float(away_goal_reg.predict(X_match)[0]))
        predicted_score_home = int(round(predicted_home_goals))
        predicted_score_away = int(round(predicted_away_goals))
        predicted_home_shots = max(0.0, float(home_shot_reg.predict(X_match)[0]))
        predicted_away_shots = max(0.0, float(away_shot_reg.predict(X_match)[0]))
        predicted_home_sot = max(0.0, float(home_sot_reg.predict(X_match)[0]))
        predicted_away_sot = max(0.0, float(away_sot_reg.predict(X_match)[0]))

        result_map = {"H": f"{home_team} win", "D": "Draw", "A": f"{away_team} win"}
        print("\nPrediction")
        print(f"Home: {home_team}")
        print(f"Away: {away_team}")
        print(f"Most likely result: {result_map[prediction]}")
        print(
            "Win probabilities: "
            f"{home_team} {probabilities['H'] * 100:.1f}% | "
            f"Draw {probabilities['D'] * 100:.1f}% | "
            f"{away_team} {probabilities['A'] * 100:.1f}%"
        )
        print(f"Predicted score: {home_team} {predicted_score_home} - {predicted_score_away} {away_team}")
        print("Predicted shots: "f"{home_team} {predicted_home_shots:.1f} | {away_team} {predicted_away_shots:.1f}")
        print(
            "Predicted shots on target: "
            f"{home_team} {predicted_home_sot:.1f} | {away_team} {predicted_away_sot:.1f}"
        )
        print("")
    


if __name__ == "__main__":
    main()
