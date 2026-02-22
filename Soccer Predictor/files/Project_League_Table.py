import os
from collections import defaultdict
from datetime import datetime
import random

import joblib
import pandas as pd

import Predict_Match as pm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "Data", "Raw_Data")
OUT_DIR = os.path.join(BASE_DIR, "Data", "Predictions")
OUT_TABLE = os.path.join(OUT_DIR, "projected_league_tables.csv")
OUT_MATCHES = os.path.join(OUT_DIR, "projected_future_matches.csv")
RNG = random.Random()


def latest_raw_file_per_competition(raw_root):
    latest = {}
    for root, _, files in os.walk(raw_root):
        for name in files:
            if not name.endswith(".csv"):
                continue
            start_year = pm.parse_season_start_year(name)
            if start_year is None:
                continue
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, raw_root)
            competition = os.path.dirname(rel_path).replace("\\", "/") or "Unknown"
            current = latest.get(competition)
            if current is None or start_year > current[0]:
                latest[competition] = (start_year, full_path)
    return {comp: path for comp, (_, path) in latest.items()}


def load_context():
    matches, season_files = pm.load_training_matches(pm.PROCESSED_DIR)
    if not os.path.exists(pm.MODEL_CACHE):
        raise FileNotFoundError(f"Missing model cache: {pm.MODEL_CACHE}. Run Predict_Match.py first.")

    bundle = joblib.load(pm.MODEL_CACHE)
    if bundle.get("fingerprint") != pm.data_fingerprint(season_files):
        raise RuntimeError("Model cache is stale. Rebuild by running Predict_Match.py.")

    overall_teams = pm.load_json_if_exists(os.path.join(pm.TEAM_DATA_DIR, "overall_teams.json"))
    season_teams = pm.load_json_if_exists(os.path.join(pm.TEAM_DATA_DIR, "season_teams.json"))
    head_to_head = pm.load_json_if_exists(os.path.join(pm.TEAM_DATA_DIR, "head_to_head.json"))
    current_form = pm.load_json_if_exists(os.path.join(pm.TEAM_DATA_DIR, "current_form.json"))
    league_strength = pm.load_json_if_exists(os.path.join(pm.TEAM_DATA_DIR, "league_strength.json")) or {}
    dynamic_form = pm.build_dynamic_form_from_matches(matches)

    if (
        overall_teams is None
        or season_teams is None
        or head_to_head is None
        or current_form is None
        or not isinstance(overall_teams, dict)
        or len(overall_teams) == 0
    ):
        overall_teams, season_teams, head_to_head, current_form = pm.build_fallback_data(matches, season_files)

    overall_teams = pm.replace_nan_with_sentinel(overall_teams)
    season_teams = pm.replace_nan_with_sentinel(season_teams)
    head_to_head = pm.replace_nan_with_sentinel(head_to_head)
    current_form = pm.replace_nan_with_sentinel(current_form)
    league_strength = pm.replace_nan_with_sentinel(league_strength)
    current_form.setdefault("teams", {})
    current_form["teams"].update(dynamic_form)

    team_comp_map = {}
    for _, row in matches.iterrows():
        team_comp_map[row["HomeTeam"]] = row["competition"]
        team_comp_map[row["AwayTeam"]] = row["competition"]

    latest_start = max(pm.parse_start_year_from_key(k) for k in season_teams.keys())
    latest_season = season_files[-1].replace(".csv", "")
    available = sorted(set(matches["HomeTeam"].dropna()) | set(matches["AwayTeam"].dropna()))

    return {
        "clf": bundle["clf"],
        "result_le": bundle["result_label_encoder"],
        "home_goal_reg": bundle["home_goal_reg"],
        "away_goal_reg": bundle["away_goal_reg"],
        "train_columns": bundle["train_columns"],
        "overall_teams": overall_teams,
        "season_teams": season_teams,
        "head_to_head": head_to_head,
        "current_form": current_form,
        "league_strength": league_strength,
        "team_comp_map": team_comp_map,
        "latest_start": latest_start,
        "latest_season": latest_season,
        "available_teams": available,
    }


def init_table(teams):
    table = {}
    for t in sorted(teams):
        table[t] = {
            "P": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "GD": 0,
            "Pts": 0,
            "PlayedReal": 0,
            "PlayedPred": 0,
        }
    return table


def apply_result(table, home, away, hg, ag, is_real):
    hs = table.setdefault(home, {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0, "PlayedReal": 0, "PlayedPred": 0})
    as_ = table.setdefault(away, {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0, "PlayedReal": 0, "PlayedPred": 0})

    hs["P"] += 1
    as_["P"] += 1
    hs["GF"] += int(hg)
    hs["GA"] += int(ag)
    as_["GF"] += int(ag)
    as_["GA"] += int(hg)
    hs["GD"] = hs["GF"] - hs["GA"]
    as_["GD"] = as_["GF"] - as_["GA"]
    if is_real:
        hs["PlayedReal"] += 1
        as_["PlayedReal"] += 1
    else:
        hs["PlayedPred"] += 1
        as_["PlayedPred"] += 1

    if hg > ag:
        hs["W"] += 1
        as_["L"] += 1
        hs["Pts"] += 3
    elif ag > hg:
        as_["W"] += 1
        hs["L"] += 1
        as_["Pts"] += 3
    else:
        hs["D"] += 1
        as_["D"] += 1
        hs["Pts"] += 1
        as_["Pts"] += 1


def predict_match(ctx, home_team, away_team, competition_hint):
    prediction_season = pm.choose_season_for_teams(home_team, away_team, ctx["season_teams"], ctx["latest_season"])
    competition_key = os.path.dirname(prediction_season).replace("\\", "/") or competition_hint
    start_year = pm.parse_start_year_from_key(prediction_season)
    season_coeff = pm.season_recency_coefficient(ctx["latest_start"], start_year)
    home_comp = ctx["team_comp_map"].get(home_team, competition_key)
    away_comp = ctx["team_comp_map"].get(away_team, competition_key)

    X = pm.build_features(
        pm.build_match_input(home_team, away_team),
        prediction_season,
        competition_key,
        season_coeff,
        ctx["overall_teams"],
        ctx["season_teams"],
        ctx["head_to_head"],
        ctx["current_form"],
        ctx["league_strength"],
        home_competition_override=home_comp,
        away_competition_override=away_comp,
    )
    X = pd.get_dummies(X, columns=["competition"], dtype=float)
    X = X.reindex(columns=ctx["train_columns"], fill_value=0.0)

    probs = {"H": 0.0, "D": 0.0, "A": 0.0}
    pvals = ctx["clf"].predict_proba(X)[0]
    for idx, enc in enumerate(ctx["clf"].classes_):
        lbl = ctx["result_le"].inverse_transform([enc])[0]
        probs[lbl] = float(pvals[idx])

    labels = ["H", "D", "A"]
    weights = [max(0.0, float(probs.get(label, 0.0))) for label in labels]
    total = sum(weights)
    if total <= 0:
        pred_res = max(probs, key=probs.get)
    else:
        pred_res = RNG.choices(labels, weights=weights, k=1)[0]
    phg = max(0.0, float(ctx["home_goal_reg"].predict(X)[0]))
    pag = max(0.0, float(ctx["away_goal_reg"].predict(X)[0]))
    hg = int(round(phg))
    ag = int(round(pag))

    # Keep scoreline direction consistent with predicted result label.
    if pred_res == "H" and hg <= ag:
        hg = ag + 1
    elif pred_res == "A" and ag <= hg:
        ag = hg + 1
    elif pred_res == "D":
        ag = hg

    return pred_res, hg, ag, probs


def project_competition(ctx, competition, raw_file):
    df = pd.read_csv(raw_file).copy()
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(df.columns):
        return [], []

    df["DateParsed"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()]
    df = df.sort_values(["DateParsed", "HomeTeam", "AwayTeam"], na_position="last").reset_index(drop=True)

    teams = sorted(set(df["HomeTeam"].astype(str).str.strip()) | set(df["AwayTeam"].astype(str).str.strip()))
    table = init_table(teams)
    future_rows = []
    seen_pairs = set()

    def add_future_prediction(home, away, match_date=""):
        pred_res, phg, pag, probs = predict_match(ctx, home, away, competition)
        apply_result(table, home, away, phg, pag, is_real=False)
        future_rows.append(
            {
                "competition": competition,
                "match_date": match_date,
                "home_team": home,
                "away_team": away,
                "predicted_result": pred_res,
                "pred_home_goals": phg,
                "pred_away_goals": pag,
                "prob_home": round(probs["H"], 6),
                "prob_draw": round(probs["D"], 6),
                "prob_away": round(probs["A"], 6),
            }
        )

    for _, row in df.iterrows():
        raw_home = str(row["HomeTeam"]).strip()
        raw_away = str(row["AwayTeam"]).strip()
        home = pm.resolve_team_name(raw_home, ctx["available_teams"]) or raw_home
        away = pm.resolve_team_name(raw_away, ctx["available_teams"]) or raw_away
        seen_pairs.add((home, away))
        ftr = str(row.get("FTR", "")).strip()
        hg = pd.to_numeric(row.get("FTHG"), errors="coerce")
        ag = pd.to_numeric(row.get("FTAG"), errors="coerce")
        is_played = ftr in {"H", "D", "A"} and pd.notna(hg) and pd.notna(ag)

        if is_played:
            apply_result(table, home, away, int(hg), int(ag), is_real=True)
            continue

        add_future_prediction(
            home,
            away,
            row["DateParsed"].date().isoformat() if pd.notna(row["DateParsed"]) else "",
        )

    # If the feed only has played matches, synthesize missing league fixtures so
    # the projection represents a full double round-robin season.
    for home in teams:
        resolved_home = pm.resolve_team_name(home, ctx["available_teams"]) or home
        for away in teams:
            if home == away:
                continue
            resolved_away = pm.resolve_team_name(away, ctx["available_teams"]) or away
            if (resolved_home, resolved_away) in seen_pairs:
                continue
            seen_pairs.add((resolved_home, resolved_away))
            add_future_prediction(resolved_home, resolved_away, "")

    out_rows = []
    ranked = sorted(table.items(), key=lambda kv: (-kv[1]["Pts"], -kv[1]["GD"], -kv[1]["GF"], kv[0]))
    for pos, (team, s) in enumerate(ranked, start=1):
        out_rows.append(
            {
                "competition": competition,
                "position": pos,
                "team": team,
                **s,
            }
        )
    return out_rows, future_rows


def main():
    ctx = load_context()
    latest = latest_raw_file_per_competition(RAW_DIR)
    if not latest:
        raise ValueError(f"No raw season files found in {RAW_DIR}")

    all_tables = []
    all_future = []
    for competition, path in sorted(latest.items()):
        table_rows, future_rows = project_competition(ctx, competition, path)
        all_tables.extend(table_rows)
        all_future.extend(future_rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(all_tables).to_csv(OUT_TABLE, index=False)
    pd.DataFrame(all_future).to_csv(OUT_MATCHES, index=False)
    print(f"Projected league tables saved: {OUT_TABLE}")
    print(f"Predicted remaining matches saved: {OUT_MATCHES}")


if __name__ == "__main__":
    main()
