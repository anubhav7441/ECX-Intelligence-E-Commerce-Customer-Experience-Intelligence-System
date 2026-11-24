"""Phases 6-8, 10-11: configurable recommendation service with cold-start labels and model-signal explanations."""
from __future__ import annotations
import json, threading
from pathlib import Path
import numpy as np
import pandas as pd
from .models import RecModel, MODELS

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "processed" / "recsys"
RES = ROOT / "recsys" / "results"
_LOCK = threading.Lock()
_SVC = None
LABEL = {"popularity": "Popular", "item_cf": "Bought by similar buyers", "user_cf": "Customers like you", "als": "Matches your buying pattern",
         "content": "Similar product", "assoc": "Often bought together"}
MIN_BUYERS_FOR_CF = 3


class RecService:
    def __init__(self, cutoff=None, weights=None):
        self.inter = pd.read_parquet(D / "interactions.parquet"); self.bk = pd.read_parquet(D / "baskets.parquet")
        self.prods = pd.read_parquet(D / "products.parquet")
        self.m = RecModel(self.inter, self.bk, self.prods, cutoff=cutoff)
        wf = RES / "hybrid_weights.json"
        self.weights = weights or (json.loads(wf.read_text()) if wf.exists() else {"popularity": 1.0})
        self.name = {p: self.m.items.get_loc(p) for p in []}

    # ------------------------------------------------------------ helpers
    def _prod(self, j):
        r = self.m.prod.iloc[j]
        return dict(product_id=self.m.items[j], short_id=self.m.items[j][:8], category=r.category,
                    median_price=None if pd.isna(r.median_price) else round(float(r.median_price), 2), buyers=int(self.m.buyers[j]))

    def tier(self, n_hist):
        return "new" if n_hist == 0 else "limited" if n_hist == 1 else "established"

    def customers(self, tier=None, n=30, seed=1):
        per = np.asarray(self.m.X.sum(1)).ravel()
        mask = per >= 2 if tier == "established" else per == 1 if tier == "limited" else per >= 1
        idx = np.where(mask)[0]
        pick = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
        return [dict(customer_id=self.m.users[i], purchases=int(per[i])) for i in pick]

    def history(self, cid):
        h = self.inter[self.inter.customer_unique_id == cid].sort_values("first_time")
        pr = self.prods.drop_duplicates("product_id").set_index("product_id")
        return [dict(product_id=r.product_id, short_id=r.product_id[:8], category=pr.category.get(r.product_id, "unknown"),
                     date=str(r.first_time)[:10], spend=round(float(r.spend), 2)) for r in h.itertuples()]

    # ------------------------------------------------------------ recommendations
    def recommend(self, cid, k=10, mode="hybrid"):
        k = max(1, min(int(k), 50)); hist = self.m.history_idx(cid)
        if cid not in self.m.users: return None
        w = self.weights if mode == "hybrid" else {mode: 1.0}
        if mode != "hybrid" and mode not in MODELS: raise ValueError("unknown model")
        top, sc, parts = self.m.recommend(hist, w, k, return_parts=True)
        tier = self.tier(len(hist)); out = []
        for r, j in enumerate(top):
            contrib = {m: float(w.get(m, 0) * parts[m][r]) for m in MODELS if w.get(m, 0)}
            sig = {m: v for m, v in contrib.items() if m != "popularity" and v > 0}
            fallback = not sig
            out.append(dict(rank=r + 1, **self._prod(j), score=round(float(sc[r]), 4), fallback=fallback,
                            source=("Popularity fallback" if fallback else LABEL[max(sig, key=sig.get)]),
                            reasons=self._reasons(hist, j, contrib, fallback)))
        note = {"established": "Personalised from this customer's purchase history.",
                "limited": "Only one past product: personalised signals are weak, so popular products fill the gaps (marked as fallback).",
                "new": "No purchase history: popular products only."}[tier]
        return dict(customer_id=cid, tier=tier, note=note, mode=mode, weights=w, k=k, history=self.history(cid), recommendations=out,
                    score_note="Scores rank products for this customer. They are NOT purchase probabilities.")

    def _reasons(self, hist, j, contrib, fallback):
        m = self.m; reasons = []; item = m.items[j]
        if fallback or contrib.get("popularity", 0) > 0 and not any(contrib.get(x, 0) for x in MODELS if x != "popularity"):
            return [f"Popular fallback: {int(m.buyers[j])} customers bought it ({int(m.recent[j])} in the last 90 days). Not personalised."]
        for h in hist:
            for r in m.rule_map.get(m.items[h], []):
                if r.consequent == item:
                    reasons.append(f"Often bought together: {r.pair_count} orders contained both this and your {m.cat[h]} product {m.items[h][:8]} (confidence {r.confidence:.2f}, lift {r.lift:.1f}).")
        if contrib.get("item_cf", 0) > 0 and len(hist):
            sims = np.asarray(m.item_sim[hist][:, j].todense()).ravel(); h = hist[int(sims.argmax())]
            reasons.append(f"Buyers of your {m.cat[h]} product {m.items[h][:8]} also bought this (item similarity {sims.max():.2f}).")
        if contrib.get("user_cf", 0) > 0: reasons.append("Customers with a similar purchase history bought this.")
        if contrib.get("als", 0) > 0: reasons.append("Matches the latent buying pattern learned from customers with several purchases (matrix factorisation).")
        if contrib.get("content", 0) > 0 and len(hist):
            cs = np.asarray((m.C[hist] @ m.C[j].T).todense()).ravel() if hasattr(m.C[hist] @ m.C[j].T, "todense") else np.asarray(m.C[hist] @ m.C[j].T).ravel()
            h = hist[int(cs.argmax())]
            reasons.append(f"Similar to your {m.cat[h]} product {m.items[h][:8]} (category and size/price profile similarity {cs.max():.2f}); this one is in {m.cat[j]}.")
        if contrib.get("popularity", 0) > 0: reasons.append(f"Also popular: {int(m.buyers[j])} buyers.")
        return reasons or ["No strong signal."]

    def recommend_new(self, category=None, k=10):
        m = self.m; k = max(1, min(int(k), 50)); s = 0.5 * m.pop_score + 0.5 * m.recent_score
        if category:
            if category not in set(m.cat): return None
            s = np.where(m.cat == category, s, -1)
        top = np.argsort(-s)[:k]
        return dict(tier="new", category=category, note="New customer: no purchase history, so these are popular products" + (f" in '{category}'." if category else " overall.") + " They are not personalised.",
                    recommendations=[dict(rank=r + 1, **self._prod(j), score=round(float(s[j]), 4), fallback=True, source="Popularity fallback",
                                          reasons=[f"Popular{' in ' + category if category else ''}: {int(m.buyers[j])} buyers ({int(m.recent[j])} in the last 90 days)."]) for r, j in enumerate(top)])

    def categories(self):
        c = pd.Series(self.m.cat).value_counts()
        return [dict(category=k, products=int(v)) for k, v in c.items()]

    def search_products(self, q="", n=25):
        pop = np.argsort(-self.m.buyers)
        res = []
        for j in pop:
            if q and q.lower() not in self.m.items[j].lower() and q.lower() not in self.m.cat[j].lower(): continue
            res.append(self._prod(j))
            if len(res) >= n: break
        return res

    def similar(self, product_id, k=10):
        m = self.m; k = max(1, min(int(k), 30))
        if product_id not in m.items: return None
        j = m.items.get_loc(product_id); top, cs, cf = m.similar_items(product_id, k)
        warn = []
        if m.buyers[j] < MIN_BUYERS_FOR_CF: warn.append(f"Only {int(m.buyers[j])} buyer(s): purchase-based similarity is unreliable, so results use product details only.")
        if m.cat[j] == "unknown": warn.append("This product has no category: similarity uses size, weight and price only.")
        return dict(product=self._prod(j), warnings=warn, similar=[dict(**self._prod(t), content_similarity=round(float(c), 3), co_purchase_similarity=round(float(f), 3),
                    reasons=[f"Product-detail similarity {c:.2f} ({'same' if m.cat[t] == m.cat[j] else 'different'} category)" + (f"; buyers overlap (similarity {f:.2f})" if f > 0 else "")]) for t, c, f in zip(top, cs, cf)],
                    score_note="Similarity is a 0-1 closeness measure, not a probability.")

    def together(self, product_id, k=10):
        m = self.m; k = max(1, min(int(k), 30))
        if product_id not in m.items: return None
        item_rules = m.rules[m.rules.antecedent == product_id].sort_values(["lift", "pair_count"], ascending=False).head(k)
        out = [dict(**self._prod(m.items.get_loc(r.consequent)), pair_count=int(r.pair_count), support=round(float(r.support), 6), confidence=round(float(r.confidence), 3), lift=round(float(r.lift), 2), level="product") for r in item_rules.itertuples()]
        note = None
        if not out:
            cat = m.cat[m.items.get_loc(product_id)]
            cr = m.cat_rules[m.cat_rules.antecedent == cat].sort_values("lift", ascending=False).head(k)
            out = [dict(category=r.consequent, pair_count=int(r.pair_count), support=round(float(r.support), 6), confidence=round(float(r.confidence), 3), lift=round(float(r.lift), 2), level="category") for r in cr.itertuples()]
            note = "No product-level rule exists for this product (too few multi-product orders). Showing category-level patterns instead."
        return dict(product=self._prod(m.items.get_loc(product_id)), note=note, together=out,
                    method="Pair counts over real multi-product orders; confidence = P(B in order | A in order); lift = confidence / P(B).",
                    n_orders=int(m.n_baskets))

    def insights(self, n=15):
        m = self.m; r = m.rules.drop_duplicates(["pair_count", "support"]).sort_values(["pair_count", "lift"], ascending=False)
        bundles = [dict(a=self._prod(m.items.get_loc(x.antecedent)), b=self._prod(m.items.get_loc(x.consequent)), pair_count=int(x.pair_count), lift=round(float(x.lift), 1)) for x in
                   m.rules[m.rules.antecedent < m.rules.consequent].sort_values(["pair_count", "lift"], ascending=False).head(n).itertuples()]
        cats = m.cat_rules[m.cat_rules.antecedent < m.cat_rules.consequent].sort_values("pair_count", ascending=False).head(n)
        return dict(n_orders=int(m.n_baskets), n_product_rules=int(len(m.rules)), n_category_rules=int(len(m.cat_rules)), bundles=bundles,
                    category_pairs=[dict(a=x.antecedent, b=x.consequent, pair_count=int(x.pair_count), confidence=round(float(x.confidence), 3), lift=round(float(x.lift), 2)) for x in cats.itertuples()],
                    caveat="Only ~3% of orders contain 2+ different products, so product pairs rest on very few orders. Lift values for rare pairs are inflated; read them together with the order count.")

    @staticmethod
    def evaluation():
        p = RES / "evaluation.json"
        return json.loads(p.read_text()) if p.exists() else None


def get_service():
    global _SVC
    with _LOCK:
        if _SVC is None: _SVC = RecService()
        return _SVC


def available():
    return (D / "interactions.parquet").exists()


def explain(svc, cid, pid, mode="hybrid"):
    """Reasons for one specific (customer, product) pair using the same signals as the ranking."""
    m = svc.m
    if cid not in m.users or pid not in m.items: return None
    hist = m.history_idx(cid); j = m.items.get_loc(pid)
    if j in set(hist.tolist()): return dict(customer_id=cid, product_id=pid, reasons=["The customer already bought this product."], contributions={})
    w = svc.weights if mode == "hybrid" else {mode: 1.0}
    sc = m.scores(hist); parts = {k: float(sc[k][j] / sc[k].max()) if sc[k].max() > 0 else 0.0 for k in MODELS}
    contrib = {k: round(w.get(k, 0) * parts[k], 4) for k in MODELS if w.get(k, 0)}
    fallback = not any(v > 0 for k, v in contrib.items() if k != "popularity")
    return dict(customer_id=cid, product_id=pid, contributions=contrib, fallback=fallback, reasons=svc._reasons(hist, j, contrib, fallback),
                note="Contributions are weighted, per-model normalised ranking scores; they are not probabilities.")
