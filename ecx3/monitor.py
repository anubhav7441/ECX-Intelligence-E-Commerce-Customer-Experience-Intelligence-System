"""Monitoring engine: watches KPIs and segments, raises alerts when a rate worsens in a statistically meaningful way.

How it decides (all deterministic, no black box):
  * "current window"  = the last `window_days` days before the as-of date (default 28)
  * "baseline window" = the `baseline_days` days just before that (default 84)
  * for the whole platform and for each state, category and seller with enough orders, the current rate is compared with that
    entity's OWN baseline using a one-sided Fisher exact test (is the rate higher?). p-values are corrected for many tests (Benjamini-Hochberg).
  * an alert needs BOTH statistical evidence (adjusted p < 0.05) AND a practical size (a minimum rise in percentage points).
  * severity: critical = adjusted p < 0.01 and a rise of at least twice the minimum; warning = otherwise.
The as-of date is the newest order in the data (not today's date), so the same code works for the historical Olist data and for live data.
An alert is a signal to investigate. It is not proof of a cause. Recent windows may be incomplete (late reviews have not arrived yet)."""
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ecx3 import store

METRICS = {
    "late_rate": {"col": "is_late", "label": "Late-delivery rate", "min_diff": 0.05},
    "dissatisfaction_rate": {"col": "dissatisfied", "label": "Unhappy-customer rate", "min_diff": 0.04},
}
ENTITIES = {  # entity_type -> (column, min orders in current window, min orders in baseline window)
    "platform": (None, 200, 400),
    "state": ("customer_state", 60, 150),
    "category": ("primary_category", 40, 120),
    "seller": ("primary_seller_id", 15, 40),
}
ALPHA, CRIT_ALPHA = 0.05, 0.01


def as_of_date(d):
    return pd.to_datetime(d["order_purchase_timestamp"]).max().normalize()


def windows(d, window_days=28, baseline_days=84, as_of=None):
    """Split orders into current and baseline windows. `as_of` (a date) replays the monitor as it would have looked on that day."""
    t = pd.to_datetime(d["order_purchase_timestamp"])
    end = (pd.Timestamp(as_of).normalize() if as_of is not None else t.max().normalize()) + pd.Timedelta(days=1)
    cs = end - pd.Timedelta(days=window_days)
    bs = cs - pd.Timedelta(days=baseline_days)
    return d[(t >= cs) & (t < end)], d[(t >= bs) & (t < cs)], (cs.date().isoformat(), end.date().isoformat(), bs.date().isoformat())


def _fisher_greater(k1, n1, k0, n0):
    return stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1]


def _severity(p_adj, diff, min_diff):
    if p_adj < CRIT_ALPHA and diff >= 2 * min_diff:
        return "critical"
    return "warning"


def _fmt(x):
    return f"{x * 100:.1f}%"


def detect(d, window_days=28, baseline_days=84, as_of=None):
    """Return a list of alert dicts (not yet stored)."""
    cur, base, (cs, end, bs) = windows(d, window_days, baseline_days, as_of)
    as_of = (pd.Timestamp(end) - pd.Timedelta(days=1)).date().isoformat()
    label = f"{cs} to {as_of} vs {bs} to {cs}"
    out = []
    for mkey, m in METRICS.items():
        col = m["col"]
        for etype, (ecol, min_cur, min_base) in ENTITIES.items():
            if ecol is None:
                rows = [("all", cur, base)]
            else:
                gc = {k: v for k, v in cur.groupby(ecol)}
                gb = {k: v for k, v in base.groupby(ecol)}
                rows = [(k, gc[k], gb[k]) for k in gc if k in gb]
            cand = []
            for key, c, b in rows:
                n1, n0 = len(c), len(b)
                if n1 < min_cur or n0 < min_base:
                    continue
                k1, k0 = int(c[col].sum()), int(b[col].sum())
                r1, r0 = k1 / n1, k0 / n0
                diff = r1 - r0
                if diff < m["min_diff"] * 0.5:          # cheap pre-filter so we do not test hopeless segments
                    cand.append((key, n1, k1, r1, n0, k0, r0, diff, 1.0))
                    continue
                cand.append((key, n1, k1, r1, n0, k0, r0, diff, _fisher_greater(k1, n1, k0, n0)))
            if not cand:
                continue
            p_adj = multipletests([c[-1] for c in cand], method="fdr_bh")[1] if len(cand) > 1 else [cand[0][-1]]
            for (key, n1, k1, r1, n0, k0, r0, diff, p), pa in zip(cand, p_adj):
                if pa >= ALPHA or diff < m["min_diff"]:
                    continue
                who = "The platform" if etype == "platform" else f"{etype.capitalize()} {key}"
                out.append({
                    "dedupe_key": f"{mkey}:{etype}:{key}", "kind": "rate_increase", "metric": mkey, "entity_type": etype, "entity_id": str(key),
                    "severity": _severity(pa, diff, m["min_diff"]),
                    "title": f"{m['label']} up for {who.lower() if etype == 'platform' else who}: {_fmt(r1)} vs {_fmt(r0)}",
                    "detail": (f"{k1} of {n1} recent orders vs {k0} of {n0} in the baseline (+{diff * 100:.1f} percentage points, adjusted p={pa:.3g}). "
                               "A statistical signal to investigate, not proof of a cause. Recent windows can be incomplete."),
                    "as_of": as_of, "window_label": label, "baseline_value": float(r0), "current_value": float(r1),
                    "n_current": int(n1), "n_baseline": int(n0), "p_value": float(pa), "effect": float(diff)})
    out.sort(key=lambda a: (0 if a["severity"] == "critical" else 1, -a["effect"]))
    return out


def run_monitor(db=None, d=None, window_days=28, baseline_days=84, max_alerts=60, as_of=None):
    """Run the monitor, store alerts (de-duplicated), record the job. Returns a summary dict."""
    if d is None:
        from ecx2.common import orders
        d = orders()
    jid = store.start_job("monitor", db)
    try:
        found = detect(d, window_days, baseline_days, as_of)[:max_alerts]
        from ecx3 import drift
        found += drift.drift_alerts(d, window_days, baseline_days, as_of)
        created = updated = 0
        for a in found:
            _, new = store.upsert_alert(a, db)
            created += int(new); updated += int(not new)
        summary = {"found": len(found), "created": created, "updated": updated, "as_of": found[0]["as_of"] if found else str(as_of_date(d).date()),
                   "critical": sum(a["severity"] == "critical" for a in found), "warning": sum(a["severity"] == "warning" for a in found),
                   "drift": sum(a["kind"] == "data_drift" for a in found)}
        store.finish_job(jid, "ok", summary, db)
        return summary
    except Exception as e:                      # record the failure, then let the caller see it
        store.finish_job(jid, "failed", {"error": str(e)[:300]}, db)
        raise
