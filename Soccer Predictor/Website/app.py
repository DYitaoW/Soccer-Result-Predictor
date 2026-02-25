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
        """Store underlying ensemble members and expose shared classes."""
        self.models = models
        self.classes_ = models[0].classes_

    def predict_proba(self, X):
        """Average probability outputs from all wrapped models."""
        matrices = [model.predict_proba(X) for model in self.models]
        return sum(matrices) / len(matrices)

    def predict(self, X):
        """Predict class labels from averaged probabilities."""
        avg = self.predict_proba(X)
        idx = avg.argmax(axis=1)
        return self.classes_[idx]


WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(WEBSITE_DIR)
FILES_DIR = os.path.join(PROJECT_DIR, "files")
MLS_FILES_DIR = os.path.join(PROJECT_DIR, "MLS", "files")
WEBSITE_FILES_DIR = os.path.join(WEBSITE_DIR, "files")
ACCURACY_HISTORY_DIR = os.path.join(WEBSITE_FILES_DIR, "accuracy_history")
ACCURACY_TOTALS_FILE = os.path.join(WEBSITE_FILES_DIR, "accuracy_totals.json")
GLOBAL_UPCOMING_FILE = os.path.join(PROJECT_DIR, "Data", "Predictions", "upcoming_matchweek_predictions.csv")
MLS_UPCOMING_FILE = os.path.join(PROJECT_DIR, "MLS", "Data", "Predictions", "upcoming_matchweek_predictions.csv")
GLOBAL_PROJECTED_TABLE_FILE = os.path.join(PROJECT_DIR, "Data", "Predictions", "projected_league_tables.csv")
MLS_PROJECTED_TABLE_FILE = os.path.join(PROJECT_DIR, "MLS", "Data", "Predictions", "projected_league_tables.csv")
MLS_PROJECTED_BRACKET_FILE = os.path.join(PROJECT_DIR, "MLS", "Data", "Predictions", "projected_mls_playoff_bracket.json")
LIVE_RESULTS_UPDATER = os.path.join(FILES_DIR, "Update_Live_Prediction_Results.py")
MLS_COMPETITION = "United States/MLS"
if FILES_DIR not in sys.path:
    sys.path.insert(0, FILES_DIR)
if MLS_FILES_DIR not in sys.path:
    sys.path.insert(0, MLS_FILES_DIR)


def _load_module(module_name, file_path):
    """Dynamically import a module from a specific file path."""
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
    """Load cached model bundle and supporting team data for one predictor mode."""
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
    """Return lazily initialized prediction context for global or MLS mode."""
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
    """Run a single match prediction and return probabilities plus stat projections."""
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
        probabilities = pm.apply_home_advantage_boost(probabilities)
        probabilities = pm.reduce_draw_probability(probabilities)
        probabilities = pm.apply_probability_randomizer(probabilities, pm.MLS_RANDOMIZER_MAX_DELTA)
    else:
        probabilities = pm.reduce_draw_probability(probabilities)
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
    """Convert H/D/A code to a display winner label."""
    code = str(code).strip().upper()
    if code == "H":
        return f"{home_team}"
    if code == "A":
        return f"{away_team}"
    return "Draw"


def _format_percent_value(value):
    """Format percent values and clamp tiny non-zero values to '<1'."""
    try:
        v = float(value)
    except Exception:
        return "0"
    if 0.0 < v < 1.0:
        return "<1"
    return f"{v:.1f}"


def _compute_accuracy_stats(frame):
    """Compute aggregate accuracy counters from a predictions dataframe."""
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
    """Compute accuracy counters grouped by competition."""
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


def _load_accuracy_totals():
    """Load persistent all-time accuracy totals written by the live updater."""
    payload = _load_json_payload(ACCURACY_TOTALS_FILE)
    if not isinstance(payload, dict):
        return {"overall": {}, "by_league": {}}
    overall = payload.get("overall")
    by_league = payload.get("by_league")
    if not isinstance(overall, dict):
        overall = {}
    if not isinstance(by_league, dict):
        by_league = {}
    return {"overall": overall, "by_league": by_league}


def _build_persistent_accuracy_stats(mode, rows):
    """Build response stats by combining persistent settled totals with current pending rows."""
    totals = _load_accuracy_totals()
    by_league_all = totals.get("by_league", {})
    if mode == "mls":
        filtered = {
            str(k): v for k, v in by_league_all.items()
            if str(k).strip() == MLS_COMPETITION
        }
    else:
        filtered = {
            str(k): v for k, v in by_league_all.items()
            if str(k).strip() != MLS_COMPETITION
        }

    pending_by_league = {}
    for row in rows:
        comp = str(row.get("competition", "")).strip() or "Unknown"
        pending_by_league[comp] = pending_by_league.get(comp, 0) + 1

    league_stats = []
    comps = sorted(set(filtered.keys()) | set(pending_by_league.keys()), key=lambda name: name.lower())
    correct_sum = 0
    settled_sum = 0
    for comp in comps:
        league_payload = filtered.get(comp, {}) if isinstance(filtered.get(comp), dict) else {}
        correct_total = int(league_payload.get("correct_total", 0) or 0)
        settled_total = int(league_payload.get("total_predictions", 0) or 0)
        pending_total = int(pending_by_league.get(comp, 0))
        accuracy_pct = round((100.0 * correct_total / settled_total), 1) if settled_total else 0.0
        league_stats.append(
            {
                "competition": comp,
                "correct_total": correct_total,
                "settled_total": settled_total,
                "pending_total": pending_total,
                "total_predictions": settled_total,
                "accuracy_pct": accuracy_pct,
            }
        )
        correct_sum += correct_total
        settled_sum += settled_total

    stats = {
        "total_predictions": settled_sum,
        "settled_total": settled_sum,
        "correct_total": correct_sum,
        "pending_total": int(len(rows)),
        "accuracy_pct": round((100.0 * correct_sum / settled_sum), 1) if settled_sum else 0.0,
    }
    return stats, league_stats


def _load_upcoming_rows(csv_path):
    """Load upcoming prediction rows and attach persistent accuracy stats."""
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
    is_mls_file = os.path.normpath(csv_path) == os.path.normpath(MLS_UPCOMING_FILE)
    # Current CSV rows are upcoming/pending only. Settled accuracy is persisted in accuracy_totals.json.
    stats = _compute_accuracy_stats(frame)
    league_stats = _compute_league_accuracy_stats(frame)
    rows = []
    for _, row in frame.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        raw_date = str(row["match_date"])
        time_label = ""
        mls_dt_raw = str(row.get("match_datetime_et", "")).strip() if "match_datetime_et" in frame.columns else ""
        if is_mls_file and mls_dt_raw:
            date_val = pd.to_datetime(mls_dt_raw, utc=True, errors="coerce")
            if pd.notna(date_val):
                try:
                    date_val = date_val.tz_convert("America/New_York")
                    time_label = date_val.strftime("%I:%M %p ET").lstrip("0")
                except Exception:
                    pass
        elif is_mls_file and len(raw_date) == 10 and raw_date.count("-") == 2:
            date_val = pd.to_datetime(raw_date, errors="coerce")
        else:
            date_val = pd.to_datetime(raw_date, utc=True, errors="coerce")
            if pd.notna(date_val) and is_mls_file:
                # MLS timestamps should be presented in Eastern time.
                try:
                    date_val = date_val.tz_convert("America/New_York")
                    time_label = date_val.strftime("%I:%M %p ET").lstrip("0")
                except Exception:
                    pass
        if pd.isna(date_val):
            weekday = ""
            date_label = str(row["match_date"])
        else:
            weekday = date_val.strftime("%A")
            date_label = date_val.strftime("%B %d, %Y")
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
                "match_datetime_et": mls_dt_raw if is_mls_file else "",
                "weekday": weekday,
                "date_label": date_label,
                "time_label": time_label,
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
    mode = "mls" if is_mls_file else "global"
    persistent_stats, persistent_league_stats = _build_persistent_accuracy_stats(mode, rows)
    return rows, persistent_stats, persistent_league_stats


def _load_projected_tables(csv_path):
    """Load projected table CSV into API-ready league/table structure."""
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
    """Safely load JSON payload from disk, returning None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def run_live_results_updater():
    """Run the live-results updater script once at app startup."""
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
    """Convert league names into filesystem-safe filenames."""
    text = "".join(ch if ch.isalnum() else "_" for ch in str(name or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:120] or "unknown_league"


def _update_accuracy_history_from_csv(csv_path, source_key):
    """Append settled predictions into per-league accuracy history CSV files."""
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
    """Refresh both global and MLS accuracy history stores."""
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
    """Render the main website page with available team lists."""
    global_ctx = get_context("global")
    mls_ctx = get_context("mls")
    return render_template("index.html", teams=global_ctx.available_teams, mls_teams=mls_ctx.available_teams)


@app.get("/api/teams")
def api_teams():
    """Return selectable teams for the requested prediction mode."""
    mode = str(request.args.get("mode", "global")).strip().lower()
    if mode not in {"global", "mls"}:
        mode = "global"
    return jsonify({"teams": get_context(mode).available_teams})


@app.post("/api/predict")
def api_predict():
    """Predict a single European/global matchup from user input."""
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
    """Predict a single MLS matchup from user input."""
    payload = request.get_json(silent=True) or request.form
    home_team = str(payload.get("home_team", "")).strip()
    away_team = str(payload.get("away_team", "")).strip()
    try:
        result = _predict(home_team, away_team, mode="mls")
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "prediction": result})


@app.get("/api/h2h")
def api_h2h():
    """Return head-to-head and form data for two teams."""
    team1 = request.args.get("team1", "").strip()
    team2 = request.args.get("team2", "").strip()
    mode = request.args.get("mode", "global").strip().lower()
    
    ctx = get_context(mode)
    
    if not team1 or not team2:
        return jsonify({"ok": False, "error": "Missing teams"}), 400
        
    t1_data = ctx.overall_teams.get(team1, {})
    t2_data = ctx.overall_teams.get(team2, {})
    t1_form = ctx.current_form.get("teams", {}).get(team1, {})
    t2_form = ctx.current_form.get("teams", {}).get(team2, {})
    
    return jsonify({
        "ok": True,
        "team1_stats": t1_data,
        "team2_stats": t2_data,
        "team1_form": t1_form,
        "team2_form": t2_form,
        "h2h_data": ctx.head_to_head.get(team1, {}).get(team2),
        "h2h_data_reverse": ctx.head_to_head.get(team2, {}).get(team1)
    })


@app.get("/api/upcoming/global")
def api_upcoming_global():
    """Return upcoming global fixtures and persistent accuracy stats."""
    rows, stats, league_stats = _load_upcoming_rows(GLOBAL_UPCOMING_FILE)
    return jsonify({"ok": True, "rows": rows, "stats": stats, "league_stats": league_stats})


@app.get("/api/upcoming/mls")
def api_upcoming_mls():
    """Return upcoming MLS fixtures and persistent accuracy stats."""
    rows, stats, league_stats = _load_upcoming_rows(MLS_UPCOMING_FILE)
    return jsonify({"ok": True, "rows": rows, "stats": stats, "league_stats": league_stats})


@app.get("/api/league-tables")
def api_league_tables():
    """Return projected league tables (and MLS playoff bracket when requested)."""
    mode = str(request.args.get("mode", "global")).strip().lower()
    if mode == "mls":
        data = _load_projected_tables(MLS_PROJECTED_TABLE_FILE)
        bracket = _load_json_payload(MLS_PROJECTED_BRACKET_FILE)
        return jsonify({"ok": True, **data, "bracket": bracket})
    else:
        data = _load_projected_tables(GLOBAL_PROJECTED_TABLE_FILE)
    return jsonify({"ok": True, **data})


@app.get("/tactics")
def tactics():
    """Render the tactics whiteboard page."""
    return render_template("tactics.html")


if __name__ == "__main__":
    run_live_results_updater()
    update_accuracy_history_files()
    app.run(host="127.0.0.1", port=5000, debug=True)
