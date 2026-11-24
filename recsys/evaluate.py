"""Phase 9: chronological evaluation lab. Run: python -m recsys.evaluate
Validation (weight search): train < 2018-03-01, validate 2018-03-01..2018-05-31.
Test (reported once):        train < 2018-06-01, test 2018-06-01..end of data.
Ground truth = products a customer bought in the period that they had NOT bought before and that exist in the training catalog."""
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from .models import RecModel, MODELS, _norm_scores

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "recsys"
OUT = ROOT / "recsys" / "results"
KS = (5, 10, 20)
HYB_PARTS = ["popularity", "item_cf", "als", "content", "assoc"]


def truth(model, inter, start, end):
    te = inter[(inter.first_time >= start) & (inter.first_time < end)]
    te = te[te.customer_unique_id.isin(model.users) & te.product_id.isin(model.items)]
    out = {}
    for u, g in te.groupby("customer_unique_id"):
        hist = set(model.items[model.history_idx(u)]); new = set(g.product_id) - hist
        if new: out[u] = new
    n_cust_all = te.customer_unique_id.nunique()
    return out, n_cust_all


def metrics_for(rec_lists, truth_sets, K):
    """per-customer metric arrays"""
    P, R, AP, N = [], [], [], []
    for u, recs in rec_lists.items():
        t = truth_sets[u]; r = recs[:K]; hit = np.array([x in t for x in r], dtype=float)
        P.append(hit.sum() / K); R.append(hit.sum() / len(t))
        prec = np.cumsum(hit) / (np.arange(len(hit)) + 1)
        AP.append((prec * hit).sum() / min(len(t), K))
        dcg = (hit / np.log2(np.arange(len(hit)) + 2)).sum(); idcg = (1 / np.log2(np.arange(min(len(t), K)) + 2)).sum()
        N.append(dcg / idcg)
    return dict(P=np.array(P), R=np.array(R), AP=np.array(AP), N=np.array(N))


def cache_scores(model, users):
    c = {}
    for u in users:
        h = model.history_idx(u); sc = model.scores(h)
        c[u] = (h, {m: _norm_scores(np.asarray(sc[m], float)) for m in MODELS})
    return c


def rank(model, h, parts, w, K):
    tot = sum(w.get(m, 0) * parts[m] for m in MODELS) + 1e-9 * model.pop_score
    tot[h] = -np.inf
    top = np.argpartition(-tot, K)[:K]; top = top[np.argsort(-tot[top])]
    sig = float(sum(w.get(m, 0) * parts[m][top].sum() for m in MODELS if m != "popularity") > 0) if any(w.get(m, 0) for m in MODELS if m != "popularity") else 1.0
    return model.items[top].tolist(), sig, top


def evaluate(model, inter, start, end, weights_by_name, ks=KS, boot=1000, seed=0):
    tr, n_all = truth(model, inter, start, end); users = list(tr)
    cache = cache_scores(model, users); Kmax = max(ks); res = {}; recs_by_model = {}
    for name, w in weights_by_name.items():
        recs, sigs, tops = {}, [], []
        for u in users:
            h, parts = cache[u]; r, s, top = rank(model, h, parts, w, Kmax); recs[u] = r; sigs.append(s); tops.append(top)
        recs_by_model[name] = recs
        entry = {"signal_rate": float(np.mean(sigs))}
        for K in ks:
            m = metrics_for(recs, tr, K)
            entry[f"P@{K}"] = float(m["P"].mean()); entry[f"R@{K}"] = float(m["R"].mean()); entry[f"MAP@{K}"] = float(m["AP"].mean()); entry[f"NDCG@{K}"] = float(m["N"].mean())
        allrec = np.concatenate([t[:10] for t in tops])
        entry["coverage@10"] = float(len(set(allrec)) / model.n_items)
        entry["novelty@10"] = float(np.mean(-np.log2(np.maximum(model.buyers[allrec], 1) / model.n_users)))
        res[name] = entry
    # bootstrap CIs on NDCG@10 and Recall@10 (resample customers)
    rng = np.random.default_rng(seed); n = len(users); idx = rng.integers(0, n, (boot, n)); per = {}
    for name in weights_by_name:
        m = metrics_for(recs_by_model[name], tr, 10); per[name] = m
        for key, lab in (("N", "NDCG@10"), ("R", "R@10")):
            bs = m[key][idx].mean(1); res[name][f"{lab}_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    diff = {}
    if "hybrid" in per and "popularity" in per:
        for key, lab in (("N", "NDCG@10"), ("R", "R@10")):
            d = per["hybrid"][key] - per["popularity"][key]
            bs = d[idx].mean(1); diff[f"hybrid_minus_popularity_{lab}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return res, diff, dict(eligible_customers=n, customers_with_any_period_purchase_in_catalog=n_all,
                           truth_items=int(sum(len(v) for v in tr.values())), train_users=model.n_users, catalog_size=model.n_items)


def simplex(parts, step=0.25):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=len(parts)):
        if sum(c) == n: yield {p: v * step for p, v in zip(parts, c)}


def main():
    inter = pd.read_parquet(D / "interactions.parquet"); bk = pd.read_parquet(D / "baskets.parquet"); pr = pd.read_parquet(D / "products.parquet")
    # ---- validation: pick hybrid weights
    vm = RecModel(inter, bk, pr, cutoff="2018-03-01")
    tr, _ = truth(vm, inter, pd.Timestamp("2018-03-01"), pd.Timestamp("2018-06-01")); users = list(tr); cache = cache_scores(vm, users)
    grid = []
    for w in simplex(HYB_PARTS):
        recs = {u: rank(vm, cache[u][0], cache[u][1], w, 10)[0] for u in users}
        m = metrics_for(recs, tr, 10); grid.append((float(m["N"].mean()), float(m["R"].mean()), w))
    grid.sort(key=lambda x: (-x[0], -x[1]))
    best = grid[0][2]
    print("validation customers:", len(users), "best weights:", best, "NDCG@10:", round(grid[0][0], 4))
    # ---- test, once
    tm = RecModel(inter, bk, pr, cutoff="2018-06-01")
    test_end = inter.first_time.max() + pd.Timedelta(seconds=1)
    named = {"popularity": {"popularity": 1}, "item_cf": {"item_cf": 1}, "user_cf": {"user_cf": 1}, "als": {"als": 1},
             "content": {"content": 1}, "assoc": {"assoc": 1}, "hybrid": best}
    res, diff, info = evaluate(tm, inter, pd.Timestamp("2018-06-01"), test_end, named)
    out = dict(protocol=dict(validation=dict(train_before="2018-03-01", period="2018-03-01 to 2018-05-31", eligible_customers=len(users)),
                             test=dict(train_before="2018-06-01", period="2018-06-01 to 2018-08-29", **info),
                             ground_truth="products bought in the period that the customer had not bought before and that exist in the training catalog",
                             K_values=list(KS), weight_grid_step=0.25, grid_size=len(grid)),
               hybrid_weights_chosen_on_validation=best, validation_top5=[dict(ndcg10=g[0], recall10=g[1], weights=g[2]) for g in grid[:5]],
               test_results=res, paired_differences=diff)
    (OUT / "evaluation.json").write_text(json.dumps(out, indent=1)); (OUT / "hybrid_weights.json").write_text(json.dumps(best))
    print(json.dumps(out["protocol"], indent=1))
    for n, e in res.items(): print(f"{n:11s} P@10={e['P@10']:.4f} R@10={e['R@10']:.4f} MAP@10={e['MAP@10']:.4f} NDCG@10={e['NDCG@10']:.4f} cov={e['coverage@10']:.4f} nov={e['novelty@10']:.2f} signal={e['signal_rate']:.2f}")
    print(diff)


if __name__ == "__main__":
    main()
