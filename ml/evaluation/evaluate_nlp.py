"""Evaluate the rule-based complaint classifier against the hand-labelled sample (150 reviews).

IMPORTANT HONESTY NOTE: labels were assigned by ONE annotator (Claude, an AI assistant), blind to classifier
output and star score, after rules were frozen. They are not independent human gold labels. The CSV
`nlp_eval_sample_labeled.csv` is provided so a human can audit/relabel and re-run this script.

Run from project root:  python -m ml.evaluation.evaluate_nlp
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from ml.sentiment_analysis.review_nlp import analyse, COMPLAINT_CATEGORIES

HERE = Path(__file__).resolve().parent


def main():
    d = pd.read_csv(HERE / "nlp_eval_sample_labeled.csv")
    d["pred"] = d.review_comment_message.map(lambda t: analyse(t)["complaint_categories"])
    d["true"] = d.true_categories.str.split(";")
    cats = COMPLAINT_CATEGORIES
    Y = np.array([[int(c in t) for c in cats] for t in d.true])
    P = np.array([[int(c in p) for c in cats] for p in d.pred])
    pr, rc, f1, sup = precision_recall_fscore_support(Y, P, zero_division=0)
    per_class = pd.DataFrame({"category": cats, "precision": pr, "recall": rc, "f1": f1, "support": sup}).round(3)
    micro = precision_recall_fscore_support(Y.ravel(), P.ravel(), average="binary", zero_division=0)[:3]
    macro_f1 = float(np.mean(f1[sup > 0]))
    exact = float(np.mean((Y == P).all(axis=1)))
    # complaint-present (any non-'other') detection
    yt = (Y[:, :-1].sum(1) > 0).astype(int); yp = (P[:, :-1].sum(1) > 0).astype(int)
    pd_, rd_, fd_, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    # jaccard per review
    jac = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(d.true, d.pred)])
    res = {
        "n_reviews": int(len(d)),
        "per_class": per_class.to_dict(orient="records"),
        "micro_precision": float(micro[0]), "micro_recall": float(micro[1]), "micro_f1": float(micro[2]),
        "macro_f1_over_supported_classes": macro_f1,
        "exact_match_accuracy": exact,
        "mean_jaccard": float(jac),
        "complaint_vs_no_complaint": {"precision": float(pd_), "recall": float(rd_), "f1": float(fd_)},
        "annotator": "single annotator (Claude, AI) - not independent human gold labels",
    }
    (HERE / "nlp_eval_results.json").write_text(json.dumps(res, indent=2))
    d[["idx", "review_comment_message", "true_categories"]].assign(pred=d.pred.map(";".join)).to_csv(
        HERE / "nlp_eval_predictions.csv", index=False)
    print(per_class.to_string(index=False))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ("per_class", "annotator")})
    return res


if __name__ == "__main__":
    main()
