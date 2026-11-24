"""Makes portable copies of the 4 XGBoost satisfaction models (models/portable/). Originals are NOT changed or retrained.
Run once (Linux/any):  python scripts/export_portable_models.py"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
out = ROOT / "models" / "portable"; out.mkdir(exist_ok=True)
for tag in ("S0", "S0_R", "S1", "S1_R"):
    art = joblib.load(ROOT / "models" / f"best_{tag}.joblib")
    model = art.pop("model")
    model.get_booster().save_model(str(out / f"best_{tag}.ubj"))
    joblib.dump(art, out / f"best_{tag}.meta.joblib")
    print("exported", tag)
