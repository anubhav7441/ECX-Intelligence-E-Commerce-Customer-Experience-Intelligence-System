"""Run review intelligence over all commented reviews + train/evaluate the sentiment classifier.

Outputs
  data/processed/review_nlp.parquet            per-review complaint flags, aspect sentiment, severity, OOF sentiment probs
  models/sentiment_tfidf_lr.joblib             trained TF-IDF + Logistic Regression sentiment model
  ml/sentiment_analysis/results/nlp_results.json
Sentiment classes are derived from stars (1-2 negative, 3 neutral, 4-5 positive): a WEAK-LABEL proxy, stated as such.
Run from project root:  python -m ml.sentiment_analysis.run_review_pipeline
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.sentiment_analysis.review_nlp import analyse, COMPLAINT_CATEGORIES, ASPECTS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)


def star_to_class(s):
    return np.where(s <= 2, "negative", np.where(s == 3, "neutral", "positive"))


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "orders_analytic.parquet")
    t = df[df.has_comment][["order_id", "review_score", "review_comment_message", "is_late", "delay_days",
                            "primary_category", "primary_seller_id", "review_creation_date"]].copy()
    res = analyse_all(t)
    R = {"n_commented_reviews": int(len(res))}

    # ---------------- complaint taxonomy statistics
    neg = res[res.review_score <= 2]
    R["n_negative_commented"] = int(len(neg))
    cat_rows = []
    for c in COMPLAINT_CATEGORIES:
        f = res[f"cat_{c}"]
        cat_rows.append({
            "category": c,
            "pct_of_negative_reviews": float(neg[f"cat_{c}"].mean() * 100),
            "pct_of_all_commented": float(f.mean() * 100),
            "mean_score_when_present": float(res.loc[f == 1, "review_score"].mean()) if f.sum() else None,
            "mean_score_when_absent": float(res.loc[f == 0, "review_score"].mean()),
            "n_present": int(f.sum()),
        })
    R["complaint_categories"] = cat_rows
    # H2: delivery-related complaint vs score
    a = res.loc[res.cat_late_delivery == 1, "review_score"]; b = res.loc[res.cat_late_delivery == 0, "review_score"]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    R["H2_delivery_complaint_vs_score"] = {"n_with": int(len(a)), "n_without": int(len(b)), "mean_with": float(a.mean()),
                                           "mean_without": float(b.mean()), "p": float(p),
                                           "cliffs_delta": float(2 * u / (len(a) * len(b)) - 1)}
    # construct validity: text says late/not delivered vs actual lateness flag
    R["construct_validity_late_text_vs_late_flag"] = {
        "pct_late_orders_with_late_complaint": float(res.loc[res.is_late == 1, "cat_late_delivery"].mean() * 100),
        "pct_ontime_orders_with_late_complaint": float(res.loc[res.is_late == 0, "cat_late_delivery"].mean() * 100),
    }
    # severity & aspects
    R["severity_distribution_negative_pct"] = (neg.severity.value_counts(normalize=True) * 100).round(2).to_dict()
    asp = {}
    for a_ in ASPECTS:
        col = res[f"aspect_{a_}"]
        mentioned = col.notna()
        asp[a_] = {"pct_reviews_mentioning": float(mentioned.mean() * 100),
                   "pct_negative_among_mentions": float((col[mentioned] == "negative").mean() * 100),
                   "pct_positive_among_mentions": float((col[mentioned] == "positive").mean() * 100),
                   "pct_negative_among_negative_reviews": float((neg[f"aspect_{a_}"] == "negative").mean() * 100)}
    R["aspects"] = asp
    # delivery-aspect negativity: actually late vs on-time orders
    dn = (res.aspect_delivery == "negative")
    R["delivery_aspect_negative_late_vs_ontime"] = {"late_pct": float(dn[res.is_late == 1].mean() * 100),
                                                    "on_time_pct": float(dn[res.is_late == 0].mean() * 100)}
    # co-occurrence among negative reviews
    cats_ = COMPLAINT_CATEGORIES[:-1]
    co = pd.DataFrame({a_: [float(((neg[f'cat_{a_}'] == 1) & (neg[f'cat_{b_}'] == 1)).sum()) for b_ in cats_] for a_ in cats_}, index=cats_)
    R["cooccurrence_negative"] = co.astype(int).to_dict()

    # ---------------- sentiment classification (weak labels from stars)
    res["sent_true"] = star_to_class(res.review_score.values)
    X, y = res.norm_text.to_numpy(dtype=object), np.asarray(res.sent_true.to_numpy(dtype=object), dtype=object)
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(res)), test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    lex_pred = res.lexicon_sentiment.to_numpy(dtype=object)[ite]
    maj = pd.Series(ytr).mode()[0]
    labels = ["negative", "neutral", "positive"]
    rep = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    R["sentiment"] = {
        "label_source": "stars: 1-2 negative, 3 neutral, 4-5 positive (weak-label proxy, not human sentiment labels)",
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "tfidf_lr": {"accuracy": float(accuracy_score(yte, pred)), "macro_f1": float(f1_score(yte, pred, average="macro")),
                     "per_class": {k: {m: float(v) for m, v in rep[k].items()} for k in labels},
                     "confusion_matrix_rows_true_cols_pred": confusion_matrix(yte, pred, labels=labels).tolist()},
        "lexicon_baseline": {"accuracy": float(accuracy_score(yte, lex_pred)), "macro_f1": float(f1_score(yte, lex_pred, average="macro"))},
        "majority_baseline": {"class": maj, "accuracy": float(accuracy_score(yte, [maj] * len(yte)))},
        "labels_order": labels,
    }
    # most informative n-grams for the negative class
    vec, lr = pipe.named_steps["tfidf"], pipe.named_steps["lr"]
    fn = np.array(vec.get_feature_names_out())
    ci = list(lr.classes_).index("negative")
    top = np.argsort(lr.coef_[ci])[::-1][:25]
    R["top_negative_ngrams"] = [(str(fn[i]), round(float(lr.coef_[ci][i]), 3)) for i in top]

    # out-of-fold probabilities for all commented reviews (used only for descriptive analytics, never as a target predictor)
    oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba", n_jobs=1)
    classes = sorted(set(y))
    res["sent_p_negative"] = oof[:, classes.index("negative")]
    res["sent_model_pred"] = np.array(classes)[oof.argmax(1)]
    pipe.fit(X, y)  # final model on all data for the app
    joblib.dump(pipe, ROOT / "models" / "sentiment_tfidf_lr.joblib")

    keep = ["order_id"] + [c for c in res.columns if c.startswith(("cat_", "aspect_"))] + [
        "lexicon_sentiment", "severity", "sent_true", "sent_p_negative", "sent_model_pred", "complaint_categories_str", "norm_text"]
    res[keep].to_parquet(ROOT / "data" / "processed" / "review_nlp.parquet", index=False)
    (OUT / "nlp_results.json").write_text(json.dumps(R, indent=2, default=str))
    return R


def analyse_all(t):
    recs = [analyse(m) for m in t.review_comment_message.values]
    out = t.reset_index(drop=True).copy()
    out["norm_text"] = [r["norm_text"] for r in recs]
    out["complaint_categories_str"] = [";".join(r["complaint_categories"]) for r in recs]
    for c in COMPLAINT_CATEGORIES:
        out[f"cat_{c}"] = [int(c in r["complaint_categories"]) for r in recs]
    for a in ASPECTS:
        out[f"aspect_{a}"] = [r["aspects"][a] for r in recs]
    out["lexicon_sentiment"] = [r["lexicon_sentiment"] for r in recs]
    out["severity"] = [r["severity"] for r in recs]
    return out


if __name__ == "__main__":
    R = main()
    print(json.dumps({k: R[k] for k in ["n_commented_reviews", "n_negative_commented", "H2_delivery_complaint_vs_score",
                                           "construct_validity_late_text_vs_late_flag", "severity_distribution_negative_pct",
                                           "delivery_aspect_negative_late_vs_ontime"]}, indent=1))
    for r in R["complaint_categories"]:
        print(f"{r['category']:<18} {r['pct_of_negative_reviews']:5.1f}% of neg | mean score present {r['mean_score_when_present']:.2f} absent {r['mean_score_when_absent']:.2f}")
    print(json.dumps(R["aspects"], indent=1))
    s = R["sentiment"]
    print("TFIDF-LR acc %.3f macroF1 %.3f | lexicon acc %.3f macroF1 %.3f | majority acc %.3f" % (
        s["tfidf_lr"]["accuracy"], s["tfidf_lr"]["macro_f1"], s["lexicon_baseline"]["accuracy"], s["lexicon_baseline"]["macro_f1"], s["majority_baseline"]["accuracy"]))
    print(s["tfidf_lr"]["per_class"]); print(s["tfidf_lr"]["confusion_matrix_rows_true_cols_pred"])
    print(R["top_negative_ngrams"][:15])
