"""Experiment 5 - automated agent evaluation against INDEPENDENTLY computed gold answers (separate pandas code path).
Measures: numeric agreement, verdict agreement, evidence completeness, action-evidence linkage, tool failures, run time.
NOT measured here (needs humans): manual-vs-agent time, usefulness ratings.      Run: python -m ecx2.evaluate_agents"""
import os
import tempfile
import time
import numpy as np
from scipy import stats
from ecx2 import store
from ecx2.agents.system import Supervisor
from ecx2.common import orders, save_json

N = 30


def gold(d, sid):
    x = d[d.primary_seller_id == sid]
    n = len(x); late = float((x.delay_days > 0).mean()) if False else float(x.is_late.mean())
    neg = float(x.dissatisfied.mean())
    cr = d.groupby("primary_category").dissatisfied.mean()
    exp = float(x.primary_category.map(cr).mean())
    p = stats.binomtest(int(x.dissatisfied.sum()), n, exp, alternative="greater").pvalue
    h1 = "Supported" if p < 0.01 and neg > exp * 1.3 else "Not supported" if p >= 0.05 else "Inconclusive"
    return {"n": n, "late": late, "neg": neg, "h1": h1}


def main():
    d = orders()
    g = d.groupby("primary_seller_id").size()
    elig = g[g >= 30].index.to_numpy()
    rng = np.random.default_rng(2026)
    pick = list(rng.choice(elig, N, replace=False))
    db = os.path.join(tempfile.mkdtemp(), "eval.db")
    ok_late = ok_neg = ok_h1 = n_find = n_ev = n_act = n_act_ok = fails = 0
    secs = []
    for sid in pick:
        t = time.time()
        out = Supervisor(db).investigate("seller", sid)
        secs.append(time.time() - t)
        if not out["ok"]:
            fails += 1; continue
        rep = out["report"]; G = gold(d, sid)
        fl = [f for r in rep.reports for f in r.findings]
        late = next((f.value for f in fl if f.metric == "late_rate"), None)
        neg = next((f.value for f in fl if f.metric == "H1_excess_dissatisfaction"), None)
        h1 = next((v["verdict"] for v in rep.verdicts if v["hypothesis"].startswith("H1")), None)
        ok_late += late is not None and abs(late - G["late"]) < 1e-9
        ok_neg += neg is not None and abs(neg - G["neg"]) < 1e-9
        ok_h1 += h1 == G["h1"]
        n_find += len(fl); n_ev += sum(1 for f in fl if f.evidence_id)
        fails += sum(len(r.errors) for r in rep.reports)
        ev_ids = {e["finding_id"] for e in store.list_evidence(run_id=rep.run_id, db=db)}
        for a in rep.actions:
            n_act += 1; n_act_ok += all(i in ev_ids for i in a.evidence_ids)
    res = {"n_sellers": N, "seed": 2026, "agreement": {"late_rate": f"{ok_late}/{N}", "dissatisfied_rate": f"{ok_neg}/{N}", "verdict_H1": f"{ok_h1}/{N}"},
           "evidence_completeness": f"{n_ev}/{n_find} findings have a ledger id", "actions_proposed": n_act,
           "actions_with_valid_evidence_links": f"{n_act_ok}/{n_act}", "failures": fails,
           "mean_seconds_per_investigation": round(float(np.mean(secs)), 2),
           "note": "Gold answers come from a separate pandas/scipy implementation, so this checks that agents report tool outputs faithfully and compute consistently; it does not prove the business conclusions are right. Manual-vs-agent timing and human usefulness ratings were NOT run."}
    save_json("agent_eval.json", res)
    print(res)


if __name__ == "__main__":
    main()
