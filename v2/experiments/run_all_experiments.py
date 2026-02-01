"""
Run All V2 Experiments
======================
Master script to run all experiments sequentially.
Run this after V2 main training completes.

Experiments:
1. Different pip thresholds (5, 8, 10)
2. V1-style binary target
3. Technical indicators
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

EXP_DIR = Path(__file__).parent

log("=" * 70)
log("RUNNING ALL V2 EXPERIMENTS")
log("=" * 70)

experiments = [
    ("Exp01: Different Pip Thresholds", "exp01_different_thresholds.py"),
    ("Exp02: V1-Style Binary Target", "exp02_v1_style_target.py"),
    ("Exp03: Technical Indicators", "exp03_technical_indicators.py"),
    ("Exp04: XGBoost Baseline", "exp04_xgboost.py"),
]

results = []

for name, script in experiments:
    log(f"\n{'='*70}")
    log(f"Starting: {name}")
    log(f"{'='*70}\n")

    script_path = EXP_DIR / script
    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
        )
        status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    except Exception as e:
        status = f"ERROR: {e}"

    end_time = datetime.now()
    duration = end_time - start_time

    results.append({
        'name': name,
        'status': status,
        'duration': str(duration),
    })

    log(f"\n{name}: {status} (took {duration})")

log(f"\n{'='*70}")
log("ALL EXPERIMENTS COMPLETE")
log(f"{'='*70}")

log(f"\n{'Experiment':<40} {'Status':<15} {'Duration':<15}")
log("-" * 70)
for r in results:
    log(f"{r['name']:<40} {r['status']:<15} {r['duration']:<15}")

log(f"\nCompleted: {datetime.now()}")
