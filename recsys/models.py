"""Recommendation models (Phases 2-6). Pure numpy/scipy/scikit-learn; scores are RANKING scores, not probabilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODELS = ["popularity", "item_cf", "user_cf", "als", "content", "assoc"]


def _norm_scores(s: np.ndarray) -> np.ndarray:
    m = s.max() if s.size else 0
    return s / m if m > 0 else s


class RecModel:
    """Holds every component fitted on training data only."""

    def __init__(self, inter: pd.DataFrame, baskets: pd.DataFrame, products: pd.DataFrame, cutoff=None,
                 recent_days=90, k_neighbors=50, als_factors=16, als_reg=0.1, als_alpha=20.0, als_iters=15,
                 min_pair=2, seed=42):
        self.cutoff = pd.Timestamp(cutoff) if cutoff is not None else inter.first_time.max() + pd.Timedelta(seconds=1)
        tr = inter[inter.first_time < self.cutoff]
        bk = baskets[baskets.order_purchase_timestamp < self.cutoff]
        self.rng = np.random.default_rng(seed)
        self.users = pd.Index(tr.customer_unique_id.unique()); self.items = pd.Index(tr.product_id.unique())
        self.n_users, self.n_items = len(self.users), len(self.items)
        u = self.users.get_indexer(tr.customer_unique_id); i = self.items.get_indexer(tr.product_id)
        self.X = sp.csr_matrix((np.ones(len(tr), dtype=np.float32), (u, i)), shape=(self.n_users, self.n_items))
        self.prod = products.drop_duplicates("product_id").set_index("product_id").reindex(self.items)
        self.prod["category"] = self.prod.category.fillna("unknown")
        self.cat = self.prod.category.values
        self.buyers = np.asarray(self.X.sum(0)).ravel()
        # --- popularity (Phase 2)
        rec = tr[tr.first_time >= self.cutoff - pd.Timedelta(days=recent_days)]
        self.recent = np.bincount(self.items.get_indexer(rec.product_id), minlength=self.n_items).astype(float)
        self.pop_score = self.buyers / max(self.buyers.max(), 1)
        self.recent_score = self.recent / max(self.recent.max(), 1)
        # --- "informative" users: >=2 distinct products (single-product users carry no co-purchase signal)
        per = np.asarray(self.X.sum(1)).ravel()
        self.multi = np.where(per >= 2)[0]
        self.Xm = self.X[self.multi]
        # --- item-based CF (Phase 3): cosine over buyers (multi users) with shrinkage, top-k neighbours
        co = (self.Xm.T @ self.Xm).tocsr().astype(np.float32); co.setdiag(0); co.eliminate_zeros()
        cnt = np.asarray(self.Xm.sum(0)).ravel()
        co = co.tocoo()
        sim = co.data / (np.sqrt(cnt[co.row] * cnt[co.col]) + 1.0)   # shrinkage lambda=1 damps tiny-support pairs
        S = sp.csr_matrix((sim, (co.row, co.col)), shape=co.shape)
        self.item_sim = self._topk(S, k_neighbors)
        # --- implicit ALS (Phase 3): trained on multi users, fold-in for any user
        self.als_reg, self.als_alpha = als_reg, als_alpha
        self._fit_als(als_factors, als_iters)
        # --- content-based (Phase 4)
        self._fit_content()
        # --- association rules (Phase 5)
        self._fit_assoc(bk, min_pair)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _topk(S, k):
        S = S.tocsr(); rows, cols, vals = [], [], []
        for r in range(S.shape[0]):
            a, b = S.indptr[r], S.indptr[r + 1]
            if b == a: continue
            d, c = S.data[a:b], S.indices[a:b]
            if len(d) > k:
                top = np.argpartition(-d, k)[:k]; d, c = d[top], c[top]
            rows.append(np.full(len(d), r)); cols.append(c); vals.append(d)
        if not rows: return sp.csr_matrix(S.shape)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=S.shape)

    def _fit_als(self, f, iters):
        rng = np.random.default_rng(42)
        Xm = self.Xm.tocsr(); nu = Xm.shape[0]
        self.Y = (rng.standard_normal((self.n_items, f)) * 0.01).astype(np.float64)
        P = np.zeros((nu, f))
        XmT = Xm.T.tocsr(); reg, alpha = self.als_reg, self.als_alpha
        for _ in range(iters):
            P = self._als_side(Xm, self.Y, reg, alpha)
            self.Y = self._als_side(XmT, P, reg, alpha)
        self.f = f

    @staticmethod
    def _als_side(M, F, reg, alpha):
        f = F.shape[1]; FtF = F.T @ F + reg * np.eye(f); out = np.zeros((M.shape[0], f))
        for r in range(M.shape[0]):
            a, b = M.indptr[r], M.indptr[r + 1]
            if a == b: continue
            Fi = F[M.indices[a:b]]
            A = FtF + alpha * (Fi.T @ Fi)
            out[r] = np.linalg.solve(A, (1 + alpha) * Fi.sum(0))
        return out

    def _fold_in(self, idx):
        Fi = self.Y[idx]
        A = self.Y.T @ self.Y + self.als_reg * np.eye(self.f) + self.als_alpha * (Fi.T @ Fi)
        return np.linalg.solve(A, (1 + self.als_alpha) * Fi.sum(0))

    def _fit_content(self):
        toks = self.prod.category.str.replace("_", " ")
        tf = TfidfVectorizer().fit_transform(toks)
        num = pd.DataFrame(index=self.prod.index)
        vol = self.prod.product_length_cm * self.prod.product_height_cm * self.prod.product_width_cm
        for name, s in [("w", np.log1p(self.prod.product_weight_g)), ("v", np.log1p(vol)), ("p", np.log1p(self.prod.median_price)),
                        ("ph", self.prod.product_photos_qty), ("dl", self.prod.product_description_lenght)]:
            miss = s.isna(); num[name] = ((s.fillna(s.median()) - s.median()) / (s.std() or 1)).clip(-3, 3).values; num[name + "_miss"] = miss.astype(float).values * 0.5
        # category carries most weight; numeric attributes refine within a category
        C = sp.hstack([tf * 2.0, sp.csr_matrix(num.values * 0.25)]).tocsr()
        self.C = normalize(C).astype(np.float32)

    def _fit_assoc(self, bk, min_pair):
        bk = bk[bk.product_id.isin(self.items)]
        n_orders = bk.order_id.nunique(); self.n_baskets = n_orders
        sz = bk.groupby("order_id").product_id.transform("nunique"); multi = bk[sz >= 2]
        item_cnt = bk.groupby("product_id").order_id.nunique()
        pair_rows = []
        for _, g in multi.groupby("order_id").product_id:
            ps = sorted(set(g))
            for a in range(len(ps)):
                for b in range(a + 1, len(ps)): pair_rows.append((ps[a], ps[b]))
        pc = pd.Series(pair_rows).value_counts() if pair_rows else pd.Series(dtype=int)
        rules = []
        for (a, b), c in pc.items():
            if c < min_pair: continue
            for x, y in ((a, b), (b, a)):
                conf = c / item_cnt[x]; lift = conf / (item_cnt[y] / n_orders)
                rules.append(dict(antecedent=x, consequent=y, pair_count=int(c), support=c / n_orders, confidence=conf, lift=lift))
        self.rules = pd.DataFrame(rules, columns=["antecedent", "consequent", "pair_count", "support", "confidence", "lift"])
        self.rule_map = {}
        for r in self.rules.itertuples():
            self.rule_map.setdefault(r.antecedent, []).append(r)
        # category-level rules: pairs of categories in the same order (more data than product pairs)
        bc = bk.assign(cat=self.items[self.items.get_indexer(bk.product_id)].map(self.prod.category))
        cs = bc.groupby("order_id").cat.agg(lambda s: sorted(set(s)))
        ccount = bc.groupby("cat").order_id.nunique(); cp = {}
        for lst in cs[cs.str.len() >= 2]:
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)): cp[(lst[a], lst[b])] = cp.get((lst[a], lst[b]), 0) + 1
        crow = []
        for (a, b), c in cp.items():
            if c < 5: continue
            for x, y in ((a, b), (b, a)):
                conf = c / ccount[x]; crow.append(dict(antecedent=x, consequent=y, pair_count=c, support=c / n_orders, confidence=conf, lift=conf / (ccount[y] / n_orders)))
        self.cat_rules = pd.DataFrame(crow, columns=["antecedent", "consequent", "pair_count", "support", "confidence", "lift"])

    # ---------------------------------------------------------------- scoring
    def history_idx(self, customer_id):
        if customer_id not in self.users: return np.array([], dtype=int)
        r = self.X[self.users.get_loc(customer_id)]
        return r.indices

    def scores(self, hist) -> dict[str, np.ndarray]:
        """Per-model score vectors over all items for a customer whose training purchases are `hist` (item indices)."""
        n = self.n_items; s = {"popularity": 0.5 * self.pop_score + 0.5 * self.recent_score}
        hist = np.asarray(hist, dtype=int)
        z = np.zeros(n)
        if len(hist) == 0:
            for m in MODELS[1:]: s[m] = z
            return s
        s["item_cf"] = np.asarray(self.item_sim[hist].sum(0)).ravel()
        uv = sp.csr_matrix((np.ones(len(hist), dtype=np.float32), (np.zeros(len(hist), dtype=int), hist)), shape=(1, n))
        co = (uv @ self.Xm.T).toarray().ravel()
        den = np.sqrt(len(hist) * np.asarray(self.Xm.sum(1)).ravel()) + 1.0
        sim = co / den
        if len(sim) > 50: sim[np.argpartition(-sim, 50)[50:]] = 0
        s["user_cf"] = np.asarray(self.Xm.T @ sim).ravel() if sim.any() else z
        p = self._fold_in(hist); s["als"] = np.clip(self.Y @ p, 0, None)
        prof = np.asarray(self.C[hist].mean(0)).ravel()
        s["content"] = np.clip(self.C @ prof, 0, None)
        a = z.copy()
        for h in hist:
            for r in self.rule_map.get(self.items[h], []):
                j = self.items.get_loc(r.consequent); a[j] += r.confidence * np.log1p(r.lift)
        s["assoc"] = a
        return s

    def recommend(self, hist, weights: dict, k=10, exclude_hist=True, return_parts=False):
        sc = self.scores(hist); tot = np.zeros(self.n_items)
        parts = {m: _norm_scores(np.asarray(sc[m], dtype=float)) for m in MODELS}
        for m, w in weights.items():
            if w: tot += w * parts[m]
        tot = tot + 1e-9 * self.pop_score   # deterministic tie-break only
        if exclude_hist and len(hist): tot[np.asarray(hist, dtype=int)] = -np.inf
        top = np.argpartition(-tot, min(k, self.n_items - 1))[:k]; top = top[np.argsort(-tot[top])]
        return (top, tot[top], {m: parts[m][top] for m in MODELS}) if return_parts else (top, tot[top])

    # ---------------------------------------------------------------- similar products
    def similar_items(self, item_id, k=10):
        """Returns (item_idx, content_sim, cf_sim). Content-based works for any product in the catalog with metadata."""
        j = self.items.get_loc(item_id)
        cs = self.C @ self.C[j].T; cs = np.asarray(cs.todense()).ravel() if sp.issparse(cs) else np.asarray(cs).ravel()
        cs[j] = -1
        top = np.argpartition(-cs, k)[:k]; top = top[np.argsort(-cs[top])]
        cf = np.asarray(self.item_sim[j].todense()).ravel()
        return top, cs[top], cf[top]
