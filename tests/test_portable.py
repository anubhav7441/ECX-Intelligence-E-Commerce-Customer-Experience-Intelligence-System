"""The portable XGBoost copies must give the same predictions as the original model files (when the original can be read)."""
import warnings
import joblib, numpy as np, pandas as pd, pytest
from pathlib import Path
import ml.portable as P

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (P.PORTABLE / "best_S1_R.ubj").exists(), reason="portable models not exported")
TAGS = ["S0", "S0_R", "S1", "S1_R"]


def _sample(art):
    d = pd.read_parquet(ROOT / "data" / "processed" / "orders_with_history.parquet").head(300)
    return art["pre"].transform(d[art["num_cols"] + art["cat_cols"]])


@pytest.mark.parametrize("tag", TAGS)
def test_portable_model_gives_valid_probabilities(tag):
    """Works on every computer: the portable copy loads and predicts."""
    warnings.filterwarnings("ignore")
    art = P.load_artifact_file(ROOT / "models" / f"best_{tag}.joblib")
    p = art["model"].predict_proba(_sample(art))
    assert p.shape == (300, 2) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert art["threshold"] > 0 and len(art["feature_names"]) > 0


@pytest.mark.parametrize("tag", TAGS)
def test_portable_equals_original(tag, monkeypatch):
    warnings.filterwarnings("ignore")
    try:
        orig = joblib.load(ROOT / "models" / f"best_{tag}.joblib")
    except Exception as e:      # some Windows PCs cannot read the original; then there is nothing to compare with
        pytest.skip(f"original model cannot be read on this computer ({type(e).__name__}); portable copy is used instead")
    real_load = joblib.load

    def broken(path, *a, **k):          # pretend the original file cannot be read
        if "portable" in str(path): return real_load(path, *a, **k)
        raise RuntimeError("input stream corrupted")
    monkeypatch.setattr(P.joblib, "load", broken)
    art = P.load_artifact_file(ROOT / "models" / f"best_{tag}.joblib")
    X = _sample(orig)
    assert np.allclose(orig["model"].predict_proba(X), art["model"].predict_proba(X), atol=1e-9)
    assert art["threshold"] == orig["threshold"] and art["feature_names"] == orig["feature_names"]


def test_missing_portable_raises_original_error(tmp_path):
    (tmp_path / "best_ZZ.joblib").write_bytes(b"not a model")
    with pytest.raises(Exception):
        P.load_artifact_file(tmp_path / "best_ZZ.joblib")
