"""Train and compare Logistic Regression, Decision Tree, Random Forest, XGBoost on 4 feature sets.

Protocol (leakage-safe):
  * temporal 70/15/15 split by purchase time; shared customers removed from train/val
  * hyper-parameters fixed a priori (no test-set tuning); XGBoost early-stops on VALIDATION
  * decision threshold = max-F1 threshold on VALIDATION; model per feature set selected by VALIDATION PR-AUC
  * all numbers reported on the untouched TEST period
  * class imbalance handled with class weights / scale_pos_weight (no resampling of test data)

Run from project root:  python -m ml.prediction.train_models
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.prediction.features import (FEATURE_SETS, TARGET, add_basic_features, add_history, build_preprocessor, temporal_split)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42


def make_model(name, spw):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, class_weight="balanced", random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample",
                                      n_jobs=2, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=40,
                             tree_method="hist", n_jobs=2, random_state=SEED)
    raise ValueError(name)


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.nanargmax(f1)])


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "threshold": float(thr),
        "accuracy": float(accuracy_score(y, yhat)), "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)), "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": float(y.mean()),
    }


def paired_auc_diff(y, p1, p2, n_boot=500, seed=SEED):
    """Paired bootstrap of AUC(p1) - AUC(p2) on the same test orders."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n = len(y); d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return {"mean_diff": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_diff_le_0": float((d <= 0).mean())}


def main():
    t0 = time.time()
    df = add_basic_features(pd.read_parquet(ROOT / "data" / "processed" / "orders_analytic.parquet"))
    nlp = pd.read_parquet(ROOT / "data" / "processed" / "review_nlp.parquet")
    late_flag = nlp.set_index("order_id").cat_late_delivery
    tr, va, te, split_info = temporal_split(df)
    prior_mean, prior_neg = float(tr.review_score.mean()), float(tr[TARGET].mean())
    # history is computed on the full chronological pool but only uses reviews created before each order's purchase
    full = add_history(pd.concat([tr, va, te]), late_flag, prior_mean, prior_neg)
    tr, va, te = full.iloc[:len(tr)].copy(), full.iloc[len(tr):len(tr) + len(va)].copy(), full.iloc[len(tr) + len(va):].copy()
    ytr, yva, yte = tr[TARGET].astype(int).values, va[TARGET].astype(int).values, te[TARGET].astype(int).values
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    R = {"split": split_info, "scale_pos_weight": spw, "results": {}, "selected": {}}
    print(split_info, flush=True)

    # baselines
    base = {"majority_class": metrics(yte, np.full(len(yte), ytr.mean()), 0.5),
            "late_flag_rule (predict dissatisfied iff late)": metrics(yte, te.is_late.values.astype(float), 0.5)}
    R["baselines"] = base
    test_preds = pd.DataFrame({"order_id": te.order_id.values, "y_true": yte})
    keep_for_boot = {}

    for fs_name, (num, cat) in FEATURE_SETS.items():
        R["results"][fs_name] = {}
        best_val_pr, best_art = -1, None
        for mname in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]:
            ts = time.time()
            pre = build_preprocessor(num, cat, scale=(mname == "Logistic Regression"))
            Xtr = pre.fit_transform(tr[num + cat]); Xva = pre.transform(va[num + cat]); Xte = pre.transform(te[num + cat])
            m = make_model(mname, spw)
            if mname == "XGBoost":
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            else:
                m.fit(Xtr, ytr)
            pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            thr = best_f1_threshold(yva, pva)
            res = {"validation": metrics(yva, pva, thr), "test": metrics(yte, pte, thr), "fit_seconds": round(time.time() - ts, 1)}
            if mname == "XGBoost":
                res["best_iteration"] = int(m.best_iteration)
            R["results"][fs_name][mname] = res
            keep_for_boot[(fs_name, mname)] = pte
            print(f"[{fs_name}] {mname:<20} val PR-AUC {res['validation']['pr_auc']:.3f} | test ROC {res['test']['roc_auc']:.3f} "
                  f"PR {res['test']['pr_auc']:.3f} F1 {res['test']['f1']:.3f} ({res['fit_seconds']}s)", flush=True)
            if res["validation"]["pr_auc"] > best_val_pr:
                best_val_pr = res["validation"]["pr_auc"]
                best_art = {"pre": pre, "model": m, "threshold": thr, "model_name": mname, "feature_set": fs_name,
                            "num_cols": num, "cat_cols": cat, "feature_names": list(pre.get_feature_names_out())}
        R["selected"][fs_name] = {"model": best_art["model_name"], "selected_by": "validation PR-AUC",
                                  "test": R["results"][fs_name][best_art["model_name"]]["test"]}
        tag = fs_name.split()[0].replace("+", "_")
        joblib.dump(best_art, ROOT / "models" / f"best_{tag}.joblib")
        test_preds[f"p_{tag}"] = keep_for_boot[(fs_name, best_art["model_name"])]

    # ---------------- RQ3 / H3: paired bootstrap on XGBoost probabilities and on best models
    comp = {}
    pairs = [("S0+R checkout-time + history", "S0 checkout-time"), ("S1+R delivery-time + history", "S1 delivery-time"),
             ("S1 delivery-time", "S0 checkout-time")]
    for a, b in pairs:
        comp[f"XGBoost: {a}  minus  {b}"] = paired_auc_diff(yte, keep_for_boot[(a, "XGBoost")], keep_for_boot[(b, "XGBoost")])
        ma, mb = R["selected"][a]["model"], R["selected"][b]["model"]
        comp[f"best-per-set ({ma} vs {mb}): {a}  minus  {b}"] = paired_auc_diff(yte, keep_for_boot[(a, ma)], keep_for_boot[(b, mb)])
    R["auc_comparisons"] = comp

    # ---------------- RQ4: is XGBoost really better than the other algorithms? (paired bootstrap on TEST, per feature set)
    algo = {}
    for fs in FEATURE_SETS:
        algo[fs] = {m: paired_auc_diff(yte, keep_for_boot[(fs, "XGBoost")], keep_for_boot[(fs, m)])
                    for m in ["Logistic Regression", "Decision Tree", "Random Forest"]}
    R["algorithm_comparisons_xgboost_minus_other_auc"] = algo

    # ---------------- sensitivity: delivery-time model on reviews created on/after delivery date
    S1_name = "S1 delivery-time"
    sel = R["selected"][S1_name]["model"]
    p = keep_for_boot[(S1_name, sel)]
    mask = te.review_after_delivery.values
    R["sensitivity_S1_review_after_delivery"] = {
        "model": sel, "n": int(mask.sum()),
        "roc_auc": float(roc_auc_score(yte[mask], p[mask])), "pr_auc": float(average_precision_score(yte[mask], p[mask])),
        "n_excluded_reviews_before_delivery": int((~mask).sum()),
        "roc_auc_excluded_group": float(roc_auc_score(yte[~mask], p[~mask])) if len(set(yte[~mask])) > 1 else None}
    # S0 on same subset for context
    p0 = keep_for_boot[("S0 checkout-time", R["selected"]["S0 checkout-time"]["model"])]
    R["sensitivity_S0_review_after_delivery"] = {"roc_auc": float(roc_auc_score(yte[mask], p0[mask])),
                                                 "pr_auc": float(average_precision_score(yte[mask], p0[mask]))}

    test_preds.to_parquet(ROOT / "data" / "processed" / "test_predictions.parquet", index=False)
    full[[c for c in ["order_id"] if c in full]].to_parquet(OUT / "_tmp_ids.parquet", index=False)
    (OUT / "_tmp_ids.parquet").unlink()
    full.to_parquet(ROOT / "data" / "processed" / "orders_with_history.parquet", index=False)
    R["total_seconds"] = round(time.time() - t0, 1)
    (OUT / "ml_results.json").write_text(json.dumps(R, indent=2))
    print("DONE in", R["total_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
