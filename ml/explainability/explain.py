"""SHAP helpers shared by the analysis script and the Flask app.

TreeExplainer gives exact Shapley values for tree models. Values are in LOG-ODDS of 'dissatisfied';
contributions of one-hot columns are summed back to their original feature (e.g. all primary_category_* -> primary_category).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
_cache = {}


def load_artifact(tag):
    if tag not in _cache:
        from ml.portable import load_artifact_file
        art = load_artifact_file(ROOT / "models" / f"best_{tag}.joblib")
        art["explainer"] = shap.TreeExplainer(art["model"])
        _cache[tag] = art
    return _cache[tag]


def origin_map(feature_names, cat_cols):
    out = {}
    for f in feature_names:
        out[f] = next((c for c in cat_cols if f.startswith(c + "_")), f)
    return out


def shap_matrix(art, df):
    X = art["pre"].transform(df[art["num_cols"] + art["cat_cols"]])
    sv = art["explainer"].shap_values(X)
    base = float(np.ravel(art["explainer"].expected_value)[0])
    return X, np.asarray(sv), base


def aggregate(sv, feature_names, cat_cols):
    om = origin_map(feature_names, cat_cols)
    parents = list(dict.fromkeys(om.values()))
    agg = np.zeros((sv.shape[0], len(parents)))
    idx = {p: i for i, p in enumerate(parents)}
    for j, f in enumerate(feature_names):
        agg[:, idx[om[f]]] += sv[:, j]
    return pd.DataFrame(agg, columns=parents)


def explain_rows(art, df, top=8):
    """Explain each row of df. Returns list of dicts: probability, prediction, base_value, contributions (top by |shap|)."""
    X, sv, base = shap_matrix(art, df)
    agg = aggregate(sv, art["feature_names"], art["cat_cols"])
    p = art["model"].predict_proba(X)[:, 1]
    res = []
    for i in range(len(df)):
        row = df.iloc[i]
        contrib = agg.iloc[i].sort_values(key=np.abs, ascending=False).head(top)
        res.append({
            "probability_dissatisfied": float(p[i]), "threshold": float(art["threshold"]),
            "predicted_dissatisfied": bool(p[i] >= art["threshold"]), "base_value_logodds": base,
            "contributions": [{"feature": f, "value": (None if pd.isna(row[f]) else (row[f].item() if hasattr(row[f], "item") else row[f])),
                               "shap_logodds": float(v), "direction": "raises risk" if v > 0 else "lowers risk"}
                              for f, v in contrib.items()],
        })
    return res
