import os
import sys
import json
import threading
import importlib.util
import subprocess
from dataclasses import dataclass

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


class AveragedProbaClassifier:
    # Cache compatibility shim for previously pickled wrappers.
    def __init__(self, models):
        self.models = models
        self.classes_ = models[0].classes_

    def predict_proba(self, X):
        matrices = [model.predict_proba(X) for model in self.models]
        return sum(matrices) / len(matrices)

    def predict(self, X):
        avg = self.predict_proba(X)
        idx = avg.argmax(axis=1)
        return self.classes_[idx]


WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(WEBSITE_DIR)
FILES_DIR = os.path.join(PROJECT_DIR, "files")
MLS_FILES_DIR = os.path.join(PROJECT_DIR, "MLS", "files")
WEBSITE_FILES_DIR = os.path.join(WEBSITE_DIR, "files")
ACCURACY_HISTORY_DIR = os.path.join(WEBSITE_FILES_DIR, "accuracy_history")
GLOBAL_UPCOMING_FILE = os.path.join(PROJECT_DIR, "Data", "Predictions", "upcoming_matchweek_predictions.csv")
MLS_UPCOMING_FILE = os.path.join(PROJECT_DIR, "MLS", "Data", "Predictions", "upcoming_matchweek_predictions.csv")
GLOBAL_PROJECTED_TABLE_FILE = os.path.join(PROJECT_DIR, "Data", "Predictions", "projected_league_tables.csv")
MLS_PROJECTED_TABLE_FILE = os.path.join(PROJECT_DIR, "MLS", "Data", "Predictions", "projected_league_tables.csv")
MLS_PROJECTED_BRACKET_FILE = os.path.join(PROJECT_DIR, "MLS", "Data", "Predictions", "projected_mls_playoff_bracket.json")
LIVE_RESULTS_UPDATER = os.path.join(FILES_DIR, "Update_Live_Prediction_Results.py")
if FILES_DIR not in sys.path:
    sys.path.insert(0, FILES_DIR)
if MLS_FILES_DIR not in sys.path:
    sys.path.insert(0, MLS_FILES_DIR)


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pm_global = _load_module("predict_match_global", os.path.join(FILES_DIR, "Predict_Match.py"))
pm_mls = _load_module("predict_match_mls", os.path.join(MLS_FILES_DIR, "Predict_Match.py"))


app = Flask(__name__, template_folder="templates", static_folder="static")


@dataclass
class PredictorContext:
    pm: object
    clf: object
    result_label_encoder: object
    home_goal_reg: object
    away_goal_reg: object
    home_shot_reg: object
    away_shot_reg: object
    home_sot_reg: object
    away_sot_reg: object
    train_columns: pd.Index
    overall_teams: dict
    season_teams: dict
    head_to_head: dict
    current_form: dict
    league_strength: dict
    latest_season: str
    latest_start_year: int
    team_competition_map: dict
    available_teams: list


_ctx_lock = threading.Lock()
_ctx_global = None
_ctx_mls = None


def _load_context(pm_mod):
    matches, season_files = pm_mod.load_training_matches(pm_mod.PROCESSED_DIR)

    if not os.path.exists(pm_mod.MODEL_CACHE):
        raise FileNotFoundError(
            f"Model cache not found at {pm_mod.MODEL_CACHE}. Run Predict_Match.py once first."
        )

    bundle = joblib.load(pm_mod.MODEL_CACHE)
    fingerprint = pm_mod.data_fingerprint(season_files)
    if bundle.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Model cache is stale for current processed data. Rebuild by running Predict_Match.py."
        )

    overall_teams = pm_mod.load_json_if_exists(os.path.join(pm_mod.TEAM_DATA_DIR, "overall_teams.json"))
    season_teams = pm_mod.load_json_if_exists(os.path.join(pm_mod.TEAM_DATA_DIR, "season_teams.json"))
    head_to_head = pm_mod.load_json_if_exists(os.path.join(pm_mod.TEAM_DATA_DIR, "head_to_head.json"))
    current_form = pm_mod.load_json_if_exists(os.path.join(pm_mod.TEAM_DATA_DIR, "current_form.json"))
    league_strength = pm_mod.load_json_if_exists(os.path.join(pm_mod.TEAM_DATA_DIR, "league_strength.json")) or {}
    dynamic_form = pm_mod.build_dynamic_form_from_matches(matches)

    if (
        overall_teams is None
        or season_teams is None
        or head_to_head is None
        or current_form is None
        or not isinstance(overall_teams, dict)
        or len(overall_teams) == 0
    ):
        overall_teams, season_teams, head_to_head, current_form = pm_mod.build_fallback_data(matches, season_files)

    overall_teams = pm_mod.replace_nan_with_sentinel(overall_teams)
    season_teams = pm_mod.replace_nan_with_sentinel(season_teams)
    head_to_head = pm_mod.replace_nan_with_sentinel(head_to_head)
    current_form = pm_mod.replace_nan_with_sentinel(current_form)
    league_strength = pm_mod.replace_nan_with_sentinel(league_strength)

    if not isinstance(current_form, dict):
        current_form = {"teams": {}}
    if "teams" not in current_form or not isinstance(current_form["teams"], dict):
        current_form["teams"] = {}
    current_form["teams"].update(dynamic_form)

    team_competition_map = {}
    for _, row in matches.iterrows():
        team_competition_map[row["HomeTeam"]] = row["competition"]
        team_competition_map[row["AwayTeam"]] = row["competition"]

    latest_season = season_files[-1].replace(".csv", "")
    latest_start_year = max(pm_mod.parse_start_year_from_key(key) for key in season_teams.keys())
    available_teams = sorted(set(matches["HomeTeam"].dropna()) | set(matches["AwayTeam"].dropna()))

    return PredictorContext(
        pm=pm_mod,
        clf=bundle["clf"],
        result_label_encoder=bundle["result_label_encoder"],
        home_goal_reg=bundle["home_goal_reg"],
        away_goal_reg=bundle["away_goal_reg"],
        home_shot_reg=bundle["home_shot_reg"],
        away_shot_reg=bundle["away_shot_reg"],
        home_sot_reg=bundle["home_sot_reg"],
        away_sot_reg=bundle["away_sot_reg"],
        train_columns=bundle["train_columns"],
        overall_teams=overall_teams,
        season_teams=season_teams,
        head_to_head=head_to_head,
        current_form=current_form,
        league_strength=league_strength,
        latest_season=latest_season,
        latest_start_year=latest_start_year,
        team_competition_map=team_competition_map,
        available_teams=available_teams,
    )


def get_context(mode="global"):
    global _ctx_global, _ctx_mls
    if mode == "mls":
        if _ctx_mls is None:
            with _ctx_lock:
                if _ctx_mls is None:
                    _ctx_mls = _load_context(pm_mls)
        return _ctx_mls

    if _ctx_global is None:
        with _ctx_lock:
            if _ctx_global is None:
                _ctx_global = _load_context(pm_global)
    return _ctx_global


def _predict(home_raw, away_raw, mode="global"):
    ctx = get_context(mode)
    pm = ctx.pm
    home_team = pm.resolve_team_name(home_raw, ctx.available_teams)
    away_team = pm.resolve_team_name(away_raw, ctx.available_teams)
    if not home_team or not away_team:
        raise ValueError("One or both team names were not recognized.")
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")

    prediction_season = pm.choose_season_for_teams(home_team, away_team, ctx.season_teams, ctx.latest_season)
    competition_key = os.path.dirname(prediction_season).replace("\\", "/") or "Unknown"
    prediction_start_year = pm.parse_start_year_from_key(prediction_season)
    season_coeff = pm.season_recency_coefficient(ctx.latest_start_year, prediction_start_year)
    home_comp = ctx.team_competition_map.get(home_team, competition_key)
    away_comp = ctx.team_competition_map.get(away_team, competition_key)

    match_input = pm.build_match_input(home_team, away_team)
    X_match = pm.build_features(
        match_input,
        prediction_season,
        competition_key,
        season_coeff,
        ctx.overall_teams,
        ctx.season_teams,
        ctx.head_to_head,
        ctx.current_form,
        ctx.league_strength,
        home_competition_override=home_comp,
        away_competition_override=away_comp,
    )
    X_match = pd.get_dummies(X_match, columns=["competition"], dtype=float)
    X_match = X_match.reindex(columns=ctx.train_columns, fill_value=0.0)

    probabilities = {"H": 0.0, "D": 0.0, "A": 0.0}
    proba_values = ctx.clf.predict_proba(X_match)[0]
    for idx, encoded_label in enumerate(ctx.clf.classes_):
        label = ctx.result_label_encoder.inverse_transform([encoded_label])[0]
        probabilities[label] = float(proba_values[idx])
    if mode == "mls":
        home_adv_shift = pm.mls_home_advantage_shift(home_team, prediction_season, ctx.season_teams)
        transfer = min(home_adv_shift, probabilities.get("A", 0.0))
        probabilities["H"] = max(0.0, probabilities.get("H", 0.0) + transfer)
        probabilities["A"] = max(0.0, probabilities.get("A", 0.0) - transfer)
        total_prob = probabilities.get("H", 0.0) + probabilities.get("D", 0.0) + probabilities.get("A", 0.0)
        if total_prob > 0:
            probabilities["H"] /= total_prob
            probabilities["D"] /= total_prob
            probabilities["A"] /= total_prob
        probabilities = pm.apply_probability_randomizer(probabilities, pm.MLS_RANDOMIZER_MAX_DELTA)
    else:
        probabilities = pm.apply_probability_randomizer(probabilities, pm.EU_RANDOMIZER_MAX_DELTA)

    prediction = max(probabilities, key=probabilities.get)
    home_goals = max(0.0, float(ctx.home_goal_reg.predict(X_match)[0]))
    away_goals = max(0.0, float(ctx.away_goal_reg.predict(X_match)[0]))
    home_shots = max(0.0, float(ctx.home_shot_reg.predict(X_match)[0]))
    away_shots = max(0.0, float(ctx.away_shot_reg.predict(X_match)[0]))
    home_sot = max(0.0, float(ctx.home_sot_reg.predict(X_match)[0]))
    away_sot = max(0.0, float(ctx.away_sot_reg.predict(X_match)[0]))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "competition": home_comp if home_comp == away_comp else f"{home_comp} vs {away_comp}",
        "predicted_result": prediction,
        "winner_label": {"H": f"{home_team} win", "D": "Draw", "A": f"{away_team} win"}[prediction],
        "prob_home": round(probabilities["H"] * 100, 3),
        "prob_draw": round(probabilities["D"] * 100, 3),
        "prob_away": round(probabilities["A"] * 100, 3),
        "pred_home_goals": int(round(home_goals)),
        "pred_away_goals": int(round(away_goals)),
        "pred_home_shots": round(home_shots, 2),
        "pred_away_shots": round(away_shots, 2),
        "pred_home_sot": round(home_sot, 2),
        "pred_away_sot": round(away_sot, 2),
    }


def _winner_label(code, home_team, away_team):
    code = str(code).strip().upper()
    if code == "H":
        return f"{home_team}"
    if code == "A":
        return f"{away_team}"
    return "Draw"


def _format_percent_value(value):
    try:
        v = float(value)
    except Exception:
        return "0"
    if 0.0 < v < 1.0:
        return "<1"
    return f"{v:.1f}"


def _compute_accuracy_stats(frame):
    if frame.empty:
        return {
            "total_predictions": 0,
            "settled_total": 0,
            "correct_total": 0,
            "pending_total": 0,
            "accuracy_pct": 0.0,
        }

    if "actual_result" in frame.columns:
        settled_mask = frame["actual_result"].astype(str).str.strip().isin({"H", "D", "A"})
    else:
        settled_mask = pd.Series([False] * len(frame), index=frame.index)
    settled = frame[settled_mask].copy()
    if settled.empty:
        return {
            "total_predictions": int(len(frame)),
            "settled_total": 0,
            "correct_total": 0,
            "pending_total": int(len(frame)),
            "accuracy_pct": 0.0,
        }

    correct = (
        settled["predicted_result"].astype(str).str.strip().str.upper()
        == settled["actual_result"].astype(str).str.strip().str.upper()
    ).sum()

    settled_total = int(len(settled))
    correct_total = int(correct)
    accuracy = round((100.0 * correct_total / settled_total), 1) if settled_total else 0.0
    return {
        "total_predictions": int(len(frame)),
        "settled_total": settled_total,
        "correct_total": correct_total,
        "pending_total": int(len(frame) - settled_total),
        "accuracy_pct": accuracy,
    }


def _compute_league_accuracy_stats(frame):
    if frame.empty or "competition" not in frame.columns:
        return []

    rows = []
    grouped = frame.groupby("competition", dropna=False)
    for competition, comp_frame in grouped:
        stats = _compute_accuracy_stats(comp_frame)
        rows.append(
            {
                "competition": str(competition),
                "correct_total": stats["correct_total"],
                "settled_total": stats["settled_total"],
                "pending_total": stats["pending_total"],
                "total_predictions": stats["total_predictions"],
                "accuracy_pct": stats["accuracy_pct"],
            }
        )
    rows.sort(key=lambda item: item["competition"].lower())
    return rows


def _load_upcoming_rows(csv_path):
    if not os.path.exists(csv_path):
        empty = pd.DataFrame()
        return [], _compute_accuracy_stats(empty), _compute_league_accuracy_stats(empty)
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        empty = pd.DataFrame()
        return [], _compute_accuracy_stats(empty), _compute_league_accuracy_stats(empty)
    if frame.empty:
        return [], _compute_accuracy_stats(frame), _compute_league_accuracy_stats(frame)

    required = ["match_date", "competition", "home_team", "away_team", "predicted_result", "prob_home", "prob_draw", "prob_away"]
    for col in required:
        if col not in frame.columns:
            return [], _compute_accuracy_stats(frame), _compute_league_accuracy_stats(frame)

    frame = frame.sort_values(["match_date", "competition", "home_team", "away_team"])
    stats = _compute_accuracy_stats(frame)
    league_stats = _compute_league_accuracy_stats(frame)
    rows = []
    for _, row in frame.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        date_val = pd.to_datetime(row["match_date"], errors="coerce")
        if pd.isna(date_val):
            weekday = ""
            date_label = str(row["match_date"])
        else:
            weekday = date_val.strftime("%A")
            date_label = date_val.strftime("%Y-%m-%d")
        try:
            ph_raw = float(row["prob_home"]) * 100
            pdv_raw = float(row["prob_draw"]) * 100
            pa_raw = float(row["prob_away"]) * 100
            ph = round(ph_raw, 3)
            pdv = round(pdv_raw, 3)
            pa = round(pa_raw, 3)
        except Exception:
            ph_raw, pdv_raw, pa_raw = 0.0, 0.0, 0.0
            ph, pdv, pa = 0.0, 0.0, 0.0
        rows.append(
            {
                "match_date": str(row["match_date"]),
                "weekday": weekday,
                "date_label": date_label,
                "competition": str(row["competition"]),
                "home_team": home,
                "away_team": away,
                "winner_label": _winner_label(row["predicted_result"], home, away),
                "prob_home": ph,
                "prob_draw": pdv,
                "prob_away": pa,
                "prob_home_text": _format_percent_value(ph_raw),
                "prob_draw_text": _format_percent_value(pdv_raw),
                "prob_away_text": _format_percent_value(pa_raw),
                "pred_home_goals": int(pd.to_numeric(row.get("pred_home_goals"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("pred_home_goals"), errors="coerce")) else None,
                "pred_away_goals": int(pd.to_numeric(row.get("pred_away_goals"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("pred_away_goals"), errors="coerce")) else None,
                "reasoning": str(row.get("probability_reasoning", "")).strip(),
                "actual_result": str(row.get("actual_result", "")).strip(),
                "is_correct": (
                    "1"
                    if str(row.get("actual_result", "")).strip().upper() in {"H", "D", "A"}
                    and str(row.get("predicted_result", "")).strip().upper()
                    == str(row.get("actual_result", "")).strip().upper()
                    else (
                        "0"
                        if str(row.get("actual_result", "")).strip().upper() in {"H", "D", "A"}
                        else ""
                    )
                ),
            }
        )
    return rows, stats, league_stats


def _load_projected_tables(csv_path):
    if not os.path.exists(csv_path):
        return {"leagues": [], "tables": {}}
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return {"leagues": [], "tables": {}}

    required = {
        "competition",
        "position",
        "team",
        "P",
        "W",
        "D",
        "L",
        "GF",
        "GA",
        "GD",
        "Pts",
    }
    if frame.empty or not required.issubset(frame.columns):
        return {"leagues": [], "tables": {}}

    frame = frame.copy()
    frame["competition"] = frame["competition"].astype(str).str.strip()
    frame = frame[frame["competition"] != ""]
    if frame.empty:
        return {"leagues": [], "tables": {}}

    frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    frame = frame.sort_values(["competition", "position", "team"], na_position="last")

    tables = {}
    for competition, comp_frame in frame.groupby("competition", dropna=False):
        rows = []
        for _, row in comp_frame.iterrows():
            rows.append(
                {
                    "position": int(row["position"]) if pd.notna(row["position"]) else 0,
                    "team": str(row["team"]),
                    "P": int(pd.to_numeric(row.get("P"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("P"), errors="coerce")) else 0,
                    "W": int(pd.to_numeric(row.get("W"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("W"), errors="coerce")) else 0,
                    "D": int(pd.to_numeric(row.get("D"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("D"), errors="coerce")) else 0,
                    "L": int(pd.to_numeric(row.get("L"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("L"), errors="coerce")) else 0,
                    "GF": int(pd.to_numeric(row.get("GF"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("GF"), errors="coerce")) else 0,
                    "GA": int(pd.to_numeric(row.get("GA"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("GA"), errors="coerce")) else 0,
                    "GD": int(pd.to_numeric(row.get("GD"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("GD"), errors="coerce")) else 0,
                    "Pts": int(pd.to_numeric(row.get("Pts"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Pts"), errors="coerce")) else 0,
                    "PlayedReal": int(pd.to_numeric(row.get("PlayedReal"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("PlayedReal"), errors="coerce")) else 0,
                    "PlayedPred": int(pd.to_numeric(row.get("PlayedPred"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("PlayedPred"), errors="coerce")) else 0,
                }
            )
        tables[str(competition)] = rows

    leagues = sorted(tables.keys(), key=lambda name: name.lower())
    return {"leagues": leagues, "tables": tables}


def _load_json_payload(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def run_live_results_updater():
    if not os.path.exists(LIVE_RESULTS_UPDATER):
        print(f"[startup] Live updater not found: {LIVE_RESULTS_UPDATER}")
        return
    try:
        proc = subprocess.run(
            [sys.executable, LIVE_RESULTS_UPDATER],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.returncode != 0:
            print("[startup] Live updater failed.")
            if proc.stderr:
                print(proc.stderr.strip())
    except Exception as exc:
        print(f"[startup] Live updater error: {exc}")


def _safe_filename(name):
    text = "".join(ch if ch.isalnum() else "_" for ch in str(name or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:120] or "unknown_league"


def _update_accuracy_history_from_csv(csv_path, source_key):
    if not os.path.exists(csv_path):
        return 0, 0
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return 0, 0
    if frame.empty or "competition" not in frame.columns or "prediction_key" not in frame.columns:
        return 0, 0

    source_dir = os.path.join(ACCURACY_HISTORY_DIR, source_key)
    os.makedirs(source_dir, exist_ok=True)
    files_touched = 0
    rows_added = 0
    history_columns = list(frame.columns)
    if "actual_result" in frame.columns:
        settled_mask = frame["actual_result"].astype(str).str.strip().isin({"H", "D", "A"})
        settled = frame[settled_mask].copy()
    else:
        settled = pd.DataFrame(columns=history_columns)

    all_competitions = sorted(set(frame["competition"].astype(str).str.strip()))
    for competition in all_competitions:
        league_name = str(competition).strip() or "Unknown"
        league_file = os.path.join(source_dir, f"{_safe_filename(league_name)}.csv")
        comp_data = settled[settled["competition"].astype(str).str.strip() == league_name].copy()
        if not comp_data.empty:
            comp_data = comp_data[history_columns].copy()
            comp_data["competition"] = league_name
        else:
            comp_data = pd.DataFrame(columns=history_columns)

        if os.path.exists(league_file):
            try:
                existing = pd.read_csv(league_file)
            except Exception:
                existing = pd.DataFrame(columns=history_columns)
        else:
            existing = pd.DataFrame(columns=history_columns)

        before = len(existing)
        merged = pd.concat([existing, comp_data], ignore_index=True) if not comp_data.empty else existing.copy()
        if not merged.empty:
            merged = merged.drop_duplicates(subset=["prediction_key"], keep="last")
            settled_mask = merged["actual_result"].astype(str).str.strip().str.upper().isin({"H", "D", "A"})
            settled = merged[settled_mask].copy()
            total_counter = int(len(settled))
            correct_counter = int(
                (
                    settled["predicted_result"].astype(str).str.strip().str.upper()
                    == settled["actual_result"].astype(str).str.strip().str.upper()
                ).sum()
            )
            accuracy_counter = round((100.0 * correct_counter / total_counter), 1) if total_counter else 0.0
            merged["correct_counter"] = correct_counter
            merged["total_counter"] = total_counter
            merged["accuracy_pct_counter"] = accuracy_counter
        else:
            merged["correct_counter"] = pd.Series(dtype="int64")
            merged["total_counter"] = pd.Series(dtype="int64")
            merged["accuracy_pct_counter"] = pd.Series(dtype="float64")
        after = len(merged)
        merged.to_csv(league_file, index=False)
        files_touched += 1
        rows_added += max(0, after - before)

    return files_touched, rows_added


def update_accuracy_history_files():
    os.makedirs(ACCURACY_HISTORY_DIR, exist_ok=True)
    global_files, global_rows = _update_accuracy_history_from_csv(GLOBAL_UPCOMING_FILE, "global")
    mls_files, mls_rows = _update_accuracy_history_from_csv(MLS_UPCOMING_FILE, "mls")
    print(
        "[startup] Accuracy history updated: "
        f"global_files={global_files}, global_new_rows={global_rows}, "
        f"mls_files={mls_files}, mls_new_rows={mls_rows}"
    )


@app.get("/")
def index():
    global_ctx = get_context("global")
    mls_ctx = get_context("mls")
    return render_template("index.html", teams=global_ctx.available_teams, mls_teams=mls_ctx.available_teams)


@app.get("/api/teams")
def api_teams():
    mode = str(request.args.get("mode", "global")).strip().lower()
    if mode not in {"global", "mls"}:
        mode = "global"
    return jsonify({"teams": get_context(mode).available_teams})


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or request.form
    home_team = str(payload.get("home_team", "")).strip()
    away_team = str(payload.get("away_team", "")).strip()
    try:
        result = _predict(home_team, away_team, mode="global")
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "prediction": result})


@app.post("/api/predict/mls")
def api_predict_mls():
    payload = request.get_json(silent=True) or request.form
    home_team = str(payload.get("home_team", "")).strip()
    away_team = str(payload.get("away_team", "")).strip()
    try:
        result = _predict(home_team, away_team, mode="mls")
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "prediction": result})


@app.get("/api/upcoming/global")
def api_upcoming_global():
    rows, stats, league_stats = _load_upcoming_rows(GLOBAL_UPCOMING_FILE)
    return jsonify({"ok": True, "rows": rows, "stats": stats, "league_stats": league_stats})


@app.get("/api/upcoming/mls")
def api_upcoming_mls():
    rows, stats, league_stats = _load_upcoming_rows(MLS_UPCOMING_FILE)
    return jsonify({"ok": True, "rows": rows, "stats": stats, "league_stats": league_stats})


@app.get("/api/league-tables")
def api_league_tables():
    mode = str(request.args.get("mode", "global")).strip().lower()
    if mode == "mls":
        data = _load_projected_tables(MLS_PROJECTED_TABLE_FILE)
        bracket = _load_json_payload(MLS_PROJECTED_BRACKET_FILE)
        return jsonify({"ok": True, **data, "bracket": bracket})
    else:
        data = _load_projected_tables(GLOBAL_PROJECTED_TABLE_FILE)
    return jsonify({"ok": True, **data})


if __name__ == "__main__":
    run_live_results_updater()
    update_accuracy_history_files()
    app.run(host="127.0.0.1", port=5000, debug=True)
