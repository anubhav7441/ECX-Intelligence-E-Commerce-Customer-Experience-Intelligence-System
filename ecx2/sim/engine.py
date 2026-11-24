"""What-if simulation engine. EVERYTHING here is HYPOTHETICAL: the delivery-time satisfaction model (v1 S1+R, XGBoost/LR artifact)
is re-scored on test-period orders after the chosen modifications. The model captures ASSOCIATIONS, not causal effects, so results are
indicative scenarios, not forecasts. Model scores are mapped to expected dissatisfied rates with isotonic calibration fitted on the
test period (monotone mapping), and uncertainty comes from bootstrapping orders."""
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from ecx2 import tools
from ecx2.common import ROOT, orders

PARAMS = {"delivery_days_reduction": (0, 30, "Reduce delivery time of every order by N days (never below 1 day)"),
          "late_orders_fixed_pct": (0, 100, "Percent of currently LATE orders that arrive exactly on the estimated date"),
          "estimate_buffer_days": (0, 30, "Add N days to the delivery estimate shown at checkout (actual delivery unchanged)"),
          "freight_discount_pct": (0, 100, "Reduce freight value by this percent")}
LABEL = "HYPOTHETICAL SCENARIO - model-based association, not a causal forecast."


@lru_cache(maxsize=1)
def _setup():
    from ml.portable import load_artifact_file
    art = load_artifact_file(ROOT / "models" / "best_S1_R.joblib")
    tp = pd.read_parquet(ROOT / "data" / "processed" / "test_predictions.parquet")
    test = orders().merge(tp[["order_id", "y_true", "p_S1_R"]], on="order_id")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(test.p_S1_R.values, test.y_true.values)
    return art, test, iso


def validate(params):
    errs, clean = [], {}
    for k, (lo, hi, _) in PARAMS.items():
        v = params.get(k, 0)
        try:
            v = float(v if v not in (None, "") else 0)
        except (TypeError, ValueError):
            errs.append(f"{k} must be a number"); continue
        if not lo <= v <= hi:
            errs.append(f"{k} must be between {lo} and {hi}")
        clean[k] = v
    unknown = set(params) - set(PARAMS) - {"seller_id", "state", "category"}
    if unknown:
        errs.append(f"unknown parameter(s): {sorted(unknown)}")
    return errs, clean


def _apply(x, p):
    """Changes are applied as DELTAS to the date-level delay (v1 definition: late = delivered date after estimated date),
    so unmodified orders keep exactly their original late flag."""
    x = x.copy()
    old_dd = x.delivery_days.copy()
    if p["late_orders_fixed_pct"] > 0:
        late = x.index[x.delay_days > 0].to_numpy()
        rng = np.random.default_rng(42)
        pick = rng.choice(late, size=int(round(len(late) * p["late_orders_fixed_pct"] / 100)), replace=False) if len(late) else []
        x.loc[pick, "delivery_days"] = (x.loc[pick, "delivery_days"] - x.loc[pick, "delay_days"]).clip(lower=1)
    if p["delivery_days_reduction"] > 0:
        x["delivery_days"] = (x.delivery_days - p["delivery_days_reduction"]).clip(lower=1)
    delta = x.delivery_days - old_dd                       # <= 0 : faster delivery
    x["estimated_days"] = x.estimated_days + p["estimate_buffer_days"]
    x["delay_days"] = x.delay_days + delta - p["estimate_buffer_days"]
    x["is_late"] = (x.delay_days > 0).astype(int)
    if p["freight_discount_pct"] > 0:
        f = 1 - p["freight_discount_pct"] / 100
        x["freight_value"] = x.freight_value * f
        x["log_freight_value"] = np.log1p(x.freight_value)
        x["freight_ratio"] = x.freight_ratio * f
    return x


def simulate(params, seller_id=None, state=None, category=None, n_boot=200):
    errs, p = validate(params)
    if errs:
        return {"ok": False, "errors": errs}
    if not any(p.values()):
        return {"ok": False, "errors": ["Choose at least one non-zero scenario parameter."]}
    art, test, iso = _setup()
    x = test
    if seller_id:
        sid, err = tools.resolve_seller(seller_id)
        if err:
            return {"ok": False, "errors": [err]}
        x = x[x.primary_seller_id == sid]
    if state:
        x = x[x.customer_state == str(state).upper()]
    if category:
        x = x[x.primary_category == category]
    if len(x) < 30:
        return {"ok": False, "errors": [f"Only {len(x)} test-period orders match the filter; need at least 30."]}
    x = x.reset_index(drop=True)
    cols = art["num_cols"] + art["cat_cols"]
    base = iso.predict(art["model"].predict_proba(art["pre"].transform(x[cols]))[:, 1])
    y = _apply(x, p)
    sim = iso.predict(art["model"].predict_proba(art["pre"].transform(y[cols]))[:, 1])
    rng = np.random.default_rng(7); d = sim - base; n = len(d)
    boots = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return {"ok": True, "label": LABEL, "cohort_n": int(n), "params": p,
            "filters": {"seller_id": seller_id, "state": state, "category": category},
            "baseline": {"late_rate": float(x.is_late.mean()), "expected_dissatisfied_rate": float(base.mean()), "observed_dissatisfied_rate": float(x.y_true.mean()),
                         "mean_delivery_days": float(x.delivery_days.mean())},
            "simulated": {"late_rate": float(y.is_late.mean()), "expected_dissatisfied_rate": float(sim.mean()), "mean_delivery_days": float(y.delivery_days.mean())},
            "change": {"late_rate_pp": float((y.is_late.mean() - x.is_late.mean()) * 100), "expected_dissatisfied_pp": float(d.mean() * 100),
                       "expected_dissatisfied_ci95_pp": [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)],
                       "avoided_dissatisfied_orders_est": float(-d.sum())},
            "assumptions": ["Other order attributes are held constant.", "The model is associative; real interventions may not reproduce these changes.",
                            "Expected rates use isotonic calibration on test-period scores.", "Interval reflects order sampling only, not model or causal uncertainty.",
                            "Test period is 2018-06 to 2018-08 (late rate lower than in 2017/early 2018)."]}
