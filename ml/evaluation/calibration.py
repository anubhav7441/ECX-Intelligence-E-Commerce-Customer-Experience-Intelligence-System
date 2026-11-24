"""Empirical meaning of the risk score: observed dissatisfied rate per score band on the untouched TEST period.
(Models use class weights, so raw scores are NOT calibrated probabilities.)  Run: python -m ml.evaluation.calibration"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    tp = pd.read_parquet(ROOT / "data" / "processed" / "test_predictions.parquet")
    out = {"note": "Observed dissatisfied rate among TEST orders whose risk score falls in each band (test base rate shown).", "models": {}}
    for tag in ["S0_R", "S1_R"]:
        p, y = tp[f"p_{tag}"].values, tp.y_true.values
        edges = np.unique(np.quantile(p, [0, .5, .8, .9, .95, 1.0]))
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi) if hi == edges[-1] else (p >= lo) & (p < hi)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "observed_dissatisfied_rate": float(y[m].mean())})
        out["models"][tag] = {"test_base_rate": float(y.mean()), "bands": bins}
    (ROOT / "ml" / "evaluation" / "calibration.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = main()
    for t, v in o["models"].items():
        print(t, "base", round(v["test_base_rate"], 3))
        for b in v["bands"]: print("  score %.2f-%.2f  n=%5d  observed %.1f%%" % (b["lo"], b["hi"], b["n"], b["observed_dissatisfied_rate"] * 100))
