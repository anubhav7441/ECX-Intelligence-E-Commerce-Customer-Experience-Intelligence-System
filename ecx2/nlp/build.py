"""Corpus-level NLP v2 outputs: per-category TF-IDF keywords (distinctive n-grams), main-issue distribution, evaluation on the
150-review labelled sample (main-issue hit rate) and multilingual-model status.   Run: python -m ecx2.nlp.build"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ecx2.common import ROOT, nlp_flags, orders, save_json
from ecx2.nlp.issues import PRIORITY, analyse_v2, v2_categories
from ml.sentiment_analysis.review_nlp import complaint_categories, normalize


def main():
    nl = nlp_flags()
    d = orders()[["order_id", "review_score", "primary_seller_id", "purchase_ym"]].merge(nl, on="order_id")
    d = d[d.norm_text.fillna("").str.split().str.len() >= 3]
    cats = [c for c in PRIORITY if c != "other"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=15, max_df=0.3, sublinear_tf=True)
    X = vec.fit_transform(d.norm_text.values)
    terms = np.array(vec.get_feature_names_out())
    kw = {}
    for c in cats:
        m = d[f"cat_{c}"].values == 1
        if m.sum() < 30:
            continue
        diff = np.asarray(X[m].mean(axis=0)).ravel() - np.asarray(X[~m].mean(axis=0)).ravel()
        kw[c] = {"n_reviews": int(m.sum()), "top_distinctive_terms": terms[np.argsort(-diff)[:12]].tolist()}
    # main issue distribution among commented dissatisfied reviews
    neg = d[d.review_score <= 2]
    main = []
    for t in neg.norm_text.values:
        cs = [c for c in cats if True]
        main.append(next((c for c in PRIORITY if c != "other" and False), None))
    # (computed via flags to avoid re-running regex on 10k texts)
    flag_cols = [f"cat_{c}" for c in PRIORITY if c != "other"]
    def pick(row):
        for c in PRIORITY:
            if c != "other" and row.get(f"cat_{c}", 0) == 1:
                return c
        return "other"
    dist = neg[flag_cols].apply(pick, axis=1).value_counts(normalize=True).round(4).to_dict() if len(neg) else {}
    # evaluation on hand-labelled sample
    ev = pd.read_csv(ROOT / "ml" / "evaluation" / "nlp_eval_sample_labeled.csv")
    hits = tot = en = 0
    for r in ev.itertuples():
        truth = set(str(r.true_categories).split(";"))
        if truth == {"other"}:
            continue
        a = analyse_v2(r.review_comment_message)
        tot += 1; hits += a["main_issue"] in truth
    def micro(fn):
        tp = fp = fn_ = 0
        for r in ev.itertuples():
            t = set(str(r.true_categories).split(";")) - {"other"}; q = set(fn(normalize(r.review_comment_message))) - {"other"}
            tp += len(t & q); fp += len(q - t); fn_ += len(t - q)
        pr, rc = tp / (tp + fp), tp / (tp + fn_)
        return {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / (pr + rc), 3)}
    res = {"micro_prf_on_labelled_150": {"v1": micro(complaint_categories), "v2_word_boundary_fix": micro(v2_categories),
                                         "note": "v1 pattern 'ma qualidade' also fired inside 'otima qualidade'; v2 requires a word start. Aggregate change on 150 reviews is negligible, but the false positive on praise is removed."},
           "keywords_by_category": kw, "main_issue_distribution_dissatisfied_with_comment": dist,
           "n_commented_reviews_used": int(len(d)),
           "main_issue_eval": {"n_labelled_with_issue": tot, "main_issue_in_true_set_rate": round(hits / tot, 3),
                               "note": "labels written by a single AI annotator (not independent human labels); small sample (150); 'hit' = predicted main issue is one of the labelled categories"},
           "multilingual_transformer_experiment": {"status": "NOT RUN", "reason": "model downloads (HuggingFace) are blocked in the build environment; no result is claimed. Script hook: ecx2/nlp/experiments.md describes the protocol."},
           "portuguese_handling": "accents/case normalised; negation window; 'praise' phrases removed before matching; English input detected and down-weighted"}
    save_json("nlp_v2_results.json", res)
    print(res["main_issue_eval"], dist)


if __name__ == "__main__":
    main()
