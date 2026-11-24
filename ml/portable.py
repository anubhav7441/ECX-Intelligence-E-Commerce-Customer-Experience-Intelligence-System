"""Loads a saved model artifact. If the original .joblib cannot be read on this computer (some Windows setups cannot
unpickle XGBoost boosters made on Linux), the portable copy in models/portable/ is used instead. The original file is never changed."""
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "models" / "portable"


def load_artifact_file(path):
    path = Path(path)
    try:
        return joblib.load(path)
    except Exception as first_error:  # XGBoostError: input stream corrupted, etc.
        meta, ubj = PORTABLE / f"{path.stem}.meta.joblib", PORTABLE / f"{path.stem}.ubj"
        if not (meta.exists() and ubj.exists()):
            raise first_error
        from xgboost import XGBClassifier
        art = joblib.load(meta)
        model = XGBClassifier()
        model.load_model(str(ubj))
        art["model"] = model
        return art
