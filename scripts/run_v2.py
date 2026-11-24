"""Rebuild every ECX-Intelligence 2.0 artifact (run AFTER the v1 pipeline: scripts/run_all.py).
Order: risk models -> anomaly detection -> NLP v2 -> agent evaluation.   python scripts/run_v2.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [("Delivery-risk models (Workflow A)", "ecx2.risk.train"), ("Anomaly detection", "ecx2.anomaly.detect"),
         ("NLP v2 keywords + evaluation", "ecx2.nlp.build"), ("Agent evaluation (Experiment 5)", "ecx2.evaluate_agents")]
for i, (name, mod) in enumerate(STEPS, 1):
    print(f"\n=== [{i}/{len(STEPS)}] {name} ===", flush=True)
    r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
    if r.returncode:
        sys.exit(f"Step failed: {mod}")
subprocess.run([sys.executable, "scripts/make_v2_status.py"], cwd=ROOT)
print("\n2.0 artifacts rebuilt.")
