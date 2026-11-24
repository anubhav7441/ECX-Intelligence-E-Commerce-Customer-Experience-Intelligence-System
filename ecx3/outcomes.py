"""Outcome measurement: did an action change the problem?

Method (difference-in-differences, the standard no-experiment approach):
  change in the entity  = rate in the `window_days` AFTER the action date minus the rate in the same number of days BEFORE it
  change in the control = the same change for all OTHER orders on the platform (shows what would have happened anyway: season, volume...)
  effect = change in entity - change in control, with a 95% confidence interval (normal approximation for proportions).
Verdict: improved / worsened only when the whole interval is on one side of zero. Otherwise "inconclusive".
Not "insufficient data" unless a window has too few orders or the after-window is mostly missing.
Caution: this is not a randomised experiment. It assumes the entity and the control would have moved in parallel without the action.
Delivery fixes also take time to reach customers, so very short windows can understate a real effect. Say this when presenting results."""
import math

import pandas as pd

from ecx3 import store
from ecx3.monitor import METRICS

MIN_N = 30
MIN_AFTER_DAYS = 14


def _entity_mask(d, etype, eid):
    col = {"state": "customer_state", "category": "primary_category", "seller": "primary_seller_id"}.get(etype)
    return pd.Series(True, index=d.index) if col is None else d[col].astype(str) == str(eid)


def _se(p, n):
    return p * (1 - p) / n if n else float("nan")


def _weekly(d, col, start, end):
    t = pd.to_datetime(d["order_purchase_timestamp"])
    wk = ((t - start).dt.days // 7)
    g = d.assign(_w=wk)[(t >= start) & (t < end)].groupby("_w")[col].agg(["mean", "size"])
    return g


def measure(d, metric, entity_type, entity_id, action_date, window_days=28):
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'")
    if window_days < 7 or window_days > 180:
        raise ValueError("window_days must be between 7 and 180")
    col = METRICS[metric]["col"]
    ad = pd.Timestamp(action_date).normalize()
    t = pd.to_datetime(d["order_purchase_timestamp"])
    last = t.max().normalize()
    b0, b1 = ad - pd.Timedelta(days=window_days), ad
    a0 = ad
    a1_full = ad + pd.Timedelta(days=window_days)
    a1 = min(a1_full, last + pd.Timedelta(days=1))
    after_days = max((a1 - a0).days, 0)
    ent = _entity_mask(d, entity_type, entity_id)
    inb, ina = (t >= b0) & (t < b1), (t >= a0) & (t < a1)
    eb, ea = d[ent & inb][col], d[ent & ina][col]
    res = {"metric": metric, "entity_type": entity_type, "entity_id": str(entity_id), "action_date": ad.date().isoformat(),
           "window_days": int(window_days), "after_days_available": int(after_days), "n_before": int(len(eb)), "n_after": int(len(ea)),
           "caution": ("Not a randomised experiment. Assumes the entity and the control would have moved in parallel without the action. "
                       "Delivery fixes take time to reach customers, so short windows can understate a real effect.")}
    if len(eb) < MIN_N or len(ea) < MIN_N or after_days < MIN_AFTER_DAYS:
        res.update(verdict="insufficient_data", reason=f"Needs at least {MIN_N} orders in each window and {MIN_AFTER_DAYS} days after the action date "
                   f"(found {len(eb)} before, {len(ea)} after, {after_days} days of data after).")
        return res
    pb, pa = float(eb.mean()), float(ea.mean())
    change_e = pa - pb
    var = _se(pb, len(eb)) + _se(pa, len(ea))
    res.update(rate_before=pb, rate_after=pa, change_entity=change_e)
    if entity_type != "platform":
        cb, ca = d[~ent & inb][col], d[~ent & ina][col]
        cpb, cpa = float(cb.mean()), float(ca.mean())
        change_c = cpa - cpb
        var += _se(cpb, len(cb)) + _se(cpa, len(ca))
        res.update(control_rate_before=cpb, control_rate_after=cpa, change_control=change_c, method="difference_in_differences")
        eff = change_e - change_c
    else:
        eff, res["method"] = change_e, "before_after_no_control"
    se = math.sqrt(var)
    lo, hi = eff - 1.96 * se, eff + 1.96 * se
    res.update(effect=eff, ci95=[lo, hi])
    res["verdict"] = "improved" if hi < 0 else "worsened" if lo > 0 else "inconclusive"
    if entity_type != "platform" and res["verdict"] == "improved" and change_e > 0:
        res["note"] = "The rate itself rose, but it rose less than everywhere else, so it did better than the control. Check that this is what the action was meant to achieve."
    elif entity_type != "platform" and res["verdict"] == "worsened" and change_e < 0:
        res["note"] = "The rate itself fell, but it fell less than everywhere else, so it did worse than the control."
    # weekly series for the chart (8 weeks before to 8 weeks after, entity vs control)
    s0 = ad - pd.Timedelta(weeks=8)
    e_w = _weekly(d[ent], col, s0, min(ad + pd.Timedelta(weeks=8), last + pd.Timedelta(days=1)))
    c_w = _weekly(d[~ent], col, s0, min(ad + pd.Timedelta(weeks=8), last + pd.Timedelta(days=1))) if entity_type != "platform" else None
    res["series"] = [{"week": int(w) - 8, "entity": float(r["mean"]), "n": int(r["size"]),
                      "control": (float(c_w.loc[w, "mean"]) if c_w is not None and w in c_w.index else None)} for w, r in e_w.iterrows() if r["size"] >= 5]
    return res


def record(d, action_id, alert_id, metric, entity_type, entity_id, action_date, window_days, actor, ops_db=None):
    r = measure(d, metric, entity_type, entity_id, action_date, window_days)
    oid = store.add_outcome(action_id, alert_id, metric, entity_type, entity_id, r["action_date"], window_days, r, r["verdict"], actor, ops_db)
    return oid, r
