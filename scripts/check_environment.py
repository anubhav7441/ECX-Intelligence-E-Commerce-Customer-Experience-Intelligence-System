"""Run:  python scripts\\check_environment.py
Prints your Python/package versions and tries to open every saved model, so problems are easy to see."""
import hashlib, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
print("Python", sys.version.split()[0], "|", sys.platform)
for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "flask"):
    try:
        import importlib.metadata as m
        print(f"  {pkg:13s}", m.version(pkg))
    except Exception as e:
        print(f"  {pkg:13s} NOT INSTALLED ({e})")
want = {"xgboost": "3.4.1", "scikit-learn": "1.9.1", "numpy": "2.5.3", "pandas": "3.0.5"}
import importlib.metadata as m
for p, v in want.items():
    try:
        if m.version(p) != v: print(f"  !! {p} is {m.version(p)} but this project was built with {v}")
    except Exception: pass
import joblib
bad = 0
for f in sorted((ROOT / "models").rglob("*.joblib")):
    h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    try:
        joblib.load(f); print(f"  OK     {f.relative_to(ROOT)}  size={f.stat().st_size}  sha256={h}")
    except Exception as e:
        bad += 1; print(f"  FAILED {f.relative_to(ROOT)}  size={f.stat().st_size}  sha256={h}  -> {type(e).__name__}: {str(e)[:120]}")
print("\nAll models load fine." if not bad else f"\n{bad} model file(s) failed. Send this whole output to your helper.")
