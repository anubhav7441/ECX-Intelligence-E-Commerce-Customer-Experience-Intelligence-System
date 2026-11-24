"""Workflow A - EARLY late-delivery risk (target: is_late) using ONLY information available before delivery.

Stage A0 (at checkout):  order, product, price, freight, distance, estimated delivery, payment, seller/state/category
                         HISTORICAL late rates computed only from orders DELIVERED strictly before this purchase.
Stage A1 (at carrier hand-off): A0 + approval_hours + seller_handling_days (known before the parcel ships).
Never used: delivery_days, delay_days, carrier_transit_days, review score/text.

Temporal 70/15/15 split; threshold = max-F1 on validation; model selection by validation PR-AUC; test untouched.
Run:  python -m ecx2.risk.train
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from ecx2.common import V2_DATA, V2_MODELS, orders, save_json
from ml.prediction.features import CAT, NUM_S0, add_basic_features, build_preprocessor
from ml.prediction.train_models import best_f1_threshold, metrics

SEED = 42
TARGET = "is_late"
HIST = ["seller_late_hist", "seller_late_hist_n", "cat_late_hist", "state_late_hist"]
PLATFORM = ["platform_late_30d", "platform_orders_7d"]
A0 = (NUM_S0 + HIST + PLATFORM, CAT)
A1 = (NUM_S0 + HIST + PLATFORM + ["approval_hours", "seller_handling_days"], CAT)
STAGES = {"A0 checkout-time": A0, "A1 carrier hand-off": A1}
K = 20


def add_late_history(d, prior):
    """Shrunk historical late rate per seller/category/state using only orders DELIVERED strictly before purchase."""
    d = d.reset_index(drop=True).copy()
    dv = pd.to_datetime(d.order_delivered_customer_date).values.astype("datetime64[ns]").astype("int64")
    pt = pd.to_datetime(d.order_purchase_timestamp).values.astype("datetime64[ns]").astype("int64")
    late = d[TARGET].values.astype(float)

    def run(key):
        out_r, out_n = np.full(len(d), prior), np.zeros(len(d))
        for _, idx in d.groupby(key).indices.items():
            o = idx[np.argsort(dv[idx], kind="stable")]
            cs = np.concatenate([[0.0], np.cumsum(late[o])])
            pos = np.searchsorted(dv[o], pt[idx], side="left")
            out_n[idx] = pos
            out_r[idx] = (cs[pos] + K * prior) / (pos + K)
        return out_r, out_n
    d["seller_late_hist"], d["seller_late_hist_n"] = run("primary_seller_id")
    d["cat_late_hist"], _ = run("primary_category")
    d["state_late_hist"], _ = run("customer_state")
    # platform load / disruption signals known at purchase time (no future information)
    o = np.argsort(dv, kind="stable"); dvs = dv[o]; cs = np.concatenate([[0.0], np.cumsum(late[o])])
    day = 86400 * 10**9
    hi = np.searchsorted(dvs, pt, side="left")            # delivered strictly before this purchase
    lo = np.searchsorted(dvs, pt - 30 * day, side="left")  # ... and within the last 30 days
    cnt = hi - lo
    d["platform_late_30d"] = np.where(cnt >= 20, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), prior)
    pts = np.sort(pt)
    d["platform_orders_7d"] = np.searchsorted(pts, pt, side="left") - np.searchsorted(pts, pt - 7 * day, side="left")
    return d


def models(spw):
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=30,
                                                class_weight="balanced_subsample", n_jobs=2, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=5, scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=40,
                                 tree_method="hist", n_jobs=2, random_state=SEED),
    }


def main():
    t0 = time.time()
    df = add_basic_features(orders()).sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df); i1, i2 = int(n * .70), int(n * .85)
    prior = float(df.iloc[:i1][TARGET].mean())
    df = add_late_history(df, prior)
    tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    te_cust = set(te.customer_unique_id)
    tr, va = tr[~tr.customer_unique_id.isin(te_cust)], va[~va.customer_unique_id.isin(te_cust)]
    ytr, yva, yte = (x[TARGET].astype(int).values for x in (tr, va, te))
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    R = {"target": TARGET, "split": {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                      "late_rate_train": float(ytr.mean()), "late_rate_val": float(yva.mean()),
                                      "late_rate_test": float(yte.mean()),
                                      "test_period": [str(te.order_purchase_timestamp.min()), str(te.order_purchase_timestamp.max())]},
         "stages": {}, "selected": {}, "baselines": {}}
    # baselines: majority, and "estimated delivery window length" only
    R["baselines"]["majority"] = metrics(yte, np.full(len(yte), ytr.mean()), 0.5)
    R["baselines"]["seller_late_history_only"] = {"roc_auc": float(roc_auc_score(yte, te.seller_late_hist)),
                                                  "pr_auc": float(average_precision_score(yte, te.seller_late_hist))}
    best_overall = {}
    for stage, (num, cat) in STAGES.items():
        R["stages"][stage] = {}
        best = (-1, None)
        for name, _ in models(spw).items():
            pre = build_preprocessor(num, cat, scale=(name == "Logistic Regression"))
            Xtr, Xva, Xte = pre.fit_transform(tr[num + cat]), pre.transform(va[num + cat]), pre.transform(te[num + cat])
            m = models(spw)[name]
            m.fit(Xtr, ytr, **({"eval_set": [(Xva, yva)], "verbose": False} if name == "XGBoost" else {}))
            pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            thr = best_f1_threshold(yva, pva)
            r = {"validation": metrics(yva, pva, thr), "test": metrics(yte, pte, thr)}
            R["stages"][stage][name] = r
            print(f"[{stage}] {name:<20} val PR {r['validation']['pr_auc']:.3f} | test ROC {r['test']['roc_auc']:.3f} "
                  f"PR {r['test']['pr_auc']:.3f} F1 {r['test']['f1']:.3f}", flush=True)
            if name != "XGBoost" and r["validation"]["pr_auc"] > best[0]:   # XGBoost is early-stopped ON validation -> its validation score is optimistic
                best = (r["validation"]["pr_auc"], dict(pre=pre, model=m, threshold=thr, model_name=name, stage=stage,
                                                        num_cols=num, cat_cols=cat, pte=pte))
        art = best[1]
        R["selected"][stage] = {"model": art["model_name"], "selected_by": "validation PR-AUC among models not early-stopped on validation (LR, RF). Rule adopted AFTER observing that XGBoost validation scores were optimistic; XGBoost results are still reported in full.",
                                "test": R["stages"][stage][art["model_name"]]["test"]}
        # observed late-rate by score decile on TEST (scores are not calibrated probabilities)
        q = pd.qcut(pd.Series(art["pte"]).rank(method="first"), 10, labels=False)
        R["selected"][stage]["test_decile_late_rate"] = [float(yte[q.values == i].mean()) for i in range(10)]
        R["selected"][stage]["top_decile_lift"] = float(yte[q.values == 9].mean() / yte.mean())
        # band cut-points from VALIDATION score quantiles (applied to test to report observed rate)
        pva = art["model"].predict_proba(art["pre"].transform(va[art["num_cols"] + art["cat_cols"]]))[:, 1]
        cuts = [float(np.quantile(pva, .70)), float(np.quantile(pva, .90))]
        band = np.where(art["pte"] < cuts[0], "Low", np.where(art["pte"] < cuts[1], "Medium", "High"))
        R["selected"][stage]["bands"] = {"cuts": cuts, "observed_test_late_rate": {
            b: {"n": int((band == b).sum()), "late_rate": float(yte[band == b].mean())} for b in ["Low", "Medium", "High"]}}
        art["bands"] = cuts
        tag = stage.split()[0]
        pte = art.pop("pte")
        joblib.dump(art, V2_MODELS / f"delivery_risk_{tag}.joblib")
        best_overall[tag] = pte
    # per-order test scores for the app (seller-level risk views)
    out = te[["order_id", "primary_seller_id", "primary_category", "customer_state", "order_purchase_timestamp", TARGET]].copy()
    out["risk_A0"], out["risk_A1"] = best_overall["A0"], best_overall["A1"]
    out.to_parquet(V2_DATA / "risk_scores_test.parquet", index=False)
    # runtime reference (the app does not ship orders_all): seller / category / state late history as of data end
    ref = {"prior_late": float(df[TARGET].mean()), "k": K,
           "seller": {s: {"rate": float(g[TARGET].mean()), "n": int(len(g))} for s, g in df.groupby("primary_seller_id")},
           "category": df.groupby("primary_category")[TARGET].mean().to_dict(),
           "state": df.groupby("customer_state")[TARGET].mean().to_dict(),
           "num_median": {c: float(df[c].median()) for c in NUM_S0 + ["approval_hours", "seller_handling_days"]},
           "platform_late_30d_latest": float(df.platform_late_30d.iloc[-1]), "platform_orders_7d_latest": float(df.platform_orders_7d.iloc[-1]),
           "seller_state": df.groupby("primary_seller_id").seller_state.first().to_dict(),
           "seller_dist_median": df.groupby("primary_seller_id").distance_km.median().to_dict()}
    save_json("risk_reference.json", ref)
    R["limitations"] = ["Analysed set = delivered orders that have a review (selection bias; undelivered/cancelled orders excluded).",
                        "Scores are not calibrated probabilities; use observed-rate bands.",
                        "Historical late rate uses only orders delivered before each purchase.",
                        "One marketplace, 2016-2018; no weather, carrier, holiday or stock data."]
    R["total_seconds"] = round(time.time() - t0, 1)
    save_json("risk_results.json", R)
    print("DONE", R["total_seconds"], "s")


if __name__ == "__main__":
    main()
