import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SP_DIR = Path(__file__).resolve().parent
ROOT_DIR = SP_DIR.parent
FILES_DIR = SP_DIR / "files"
MLS_FILES_DIR = SP_DIR / "MLS" / "files"
LOCAL_KEYS_FILE = FILES_DIR / "local_api_keys.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full soccer pipeline: data pull -> processing -> model cache -> predictions -> tables -> settle."
    )
    parser.add_argument(
        "--skip-mls",
        action="store_true",
        help="Skip MLS pipeline steps.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Skip European/global pipeline steps.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=3,
        help="Fixture window days for upcoming matchweek scripts.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps if one step fails.",
    )
    return parser.parse_args()


def load_api_token():
    env_token = os.getenv("FOOTBALL_DATA_API_TOKEN", "").strip()
    if env_token:
        return env_token
    if LOCAL_KEYS_FILE.exists():
        try:
            payload = json.loads(LOCAL_KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        token = str(payload.get("FOOTBALL_DATA_API_TOKEN", "")).strip()
        if token:
            return token
    return ""


def run_step(name, cmd, continue_on_error=False, input_text=None):
    print(f"\n=== {name} ===")
    print(" ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            input=input_text,
            check=False,
        )
    except Exception as exc:
        print(f"[ERROR] {name}: {exc}")
        if continue_on_error:
            return False
        raise

    if proc.returncode != 0:
        print(f"[ERROR] {name} failed with exit code {proc.returncode}")
        if continue_on_error:
            return False
        raise SystemExit(proc.returncode)

    print(f"[OK] {name}")
    return True


def main():
    args = parse_args()
    py = sys.executable
    api_token = load_api_token()

    if not args.skip_global:
        run_step(
            "Global download latest data",
            [py, str(FILES_DIR / "Download_Latest_Data.py")],
            continue_on_error=args.continue_on_error,
        )
        run_step(
            "Global process data",
            [py, str(FILES_DIR / "Process_Data.py")],
            continue_on_error=args.continue_on_error,
        )
        run_step(
            "Global sort data",
            [py, str(FILES_DIR / "Sort_Data.py")],
            continue_on_error=args.continue_on_error,
        )
        run_step(
            "Global build model cache (non-interactive)",
            [py, str(FILES_DIR / "Predict_Match.py")],
            continue_on_error=args.continue_on_error,
            input_text="n\nq\n",
        )
        upcoming_cmd = [py, str(FILES_DIR / "Predict_Upcoming_Matchweek.py"), "--window-days", str(args.window_days)]
        if api_token:
            upcoming_cmd += ["--api-token", api_token]
        run_step(
            "Global upcoming matchweek predictions",
            upcoming_cmd,
            continue_on_error=args.continue_on_error,
        )
        run_step(
            "Global projected league tables",
            [py, str(FILES_DIR / "Project_League_Table.py")],
            continue_on_error=args.continue_on_error,
        )

    if not args.skip_mls:
        run_step(
            "MLS download/process/sort latest data",
            [py, str(MLS_FILES_DIR / "Download_Latest_Data.py")],
            continue_on_error=args.continue_on_error,
        )
        run_step(
            "MLS build model cache (non-interactive)",
            [py, str(MLS_FILES_DIR / "Predict_Match.py")],
            continue_on_error=args.continue_on_error,
            input_text="n\nq\n",
        )
        mls_upcoming_cmd = [py, str(MLS_FILES_DIR / "Predict_Upcoming_Matchweek.py"), "--window-days", str(args.window_days)]
        if api_token:
            mls_upcoming_cmd += ["--api-token", api_token]
        run_step(
            "MLS upcoming matchweek predictions",
            mls_upcoming_cmd,
            continue_on_error=args.continue_on_error,
        )
        run_step(
            "MLS projected league tables",
            [py, str(MLS_FILES_DIR / "Project_League_Table.py")],
            continue_on_error=args.continue_on_error,
        )

    run_step(
        "Settle predictions with live/final results",
        [py, str(FILES_DIR / "Update_Live_Prediction_Results.py")],
        continue_on_error=args.continue_on_error,
    )

    # Refresh website accuracy-history store for permanent counters.
    run_step(
        "Update website accuracy history",
        [
            py,
            "-c",
            (
                "import importlib.util; "
                "p=r'Soccer Predictor/Website/app.py'; "
                "s=importlib.util.spec_from_file_location('webapp', p); "
                "m=importlib.util.module_from_spec(s); "
                "s.loader.exec_module(m); "
                "m.update_accuracy_history_files()"
            ),
        ],
        continue_on_error=args.continue_on_error,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
