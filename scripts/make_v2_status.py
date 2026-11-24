"""Write v2_results/model_registry.json and experiments.json from the measured result files (no hand-typed numbers)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecx2.common import ROOT, load_json, save_json

R = json.load(open(ROOT / "ml/prediction/results/ml_results.json"))
K = load_json("risk_results.json")
A0, A1 = K["selected"]["A0 checkout-time"], K["selected"]["A1 carrier hand-off"]
auc = lambda k: round(R["selected"][k]["test"]["roc_auc"], 3)
v1 = lambda n, k, f: {"name": n, "version": "v1", "task": "dissatisfied (1-2 stars)", "algorithm": R["selected"][k]["model"], "test_roc_auc": auc(k), "file": f}
save_json("model_registry.json", {"note": "v1 artifacts are untouched. v2 artifacts live in models/v2/.", "models": [
    v1("Satisfaction checkout-time", "S0 checkout-time", "models/best_S0.joblib"),
    v1("Satisfaction checkout + history", "S0+R checkout-time + history", "models/best_S0_R.joblib"),
    v1("Satisfaction delivery-time", "S1 delivery-time", "models/best_S1.joblib"),
    v1("Satisfaction delivery + history", "S1+R delivery-time + history", "models/best_S1_R.joblib (used by the what-if simulator)"),
    {"name": "Review sentiment", "version": "v1", "task": "sentiment (weak labels from stars)", "algorithm": "TF-IDF + Logistic Regression", "test_roc_auc": None, "file": "models/sentiment_tfidf_lr.joblib"},
    {"name": "Late-risk checkout stage (A0)", "version": "v2", "task": "is_late", "algorithm": A0["model"], "test_roc_auc": round(A0["test"]["roc_auc"], 3), "file": "models/v2/delivery_risk_A0.joblib"},
    {"name": "Late-risk carrier hand-off (A1)", "version": "v2", "task": "is_late", "algorithm": A1["model"], "test_roc_auc": round(A1["test"]["roc_auc"], 3), "file": "models/v2/delivery_risk_A1.joblib"},
    {"name": "Seller anomaly ranking", "version": "v2", "task": "unsupervised", "algorithm": "Isolation Forest (+ FDR-controlled tests)", "test_roc_auc": None, "file": "computed by ecx2/anomaly/detect.py (no stored model)"}]})
E = lambda i, t, s, d: {"id": i, "title": t, "status": s, "detail": d}
save_json("experiments.json", {"items": [
    E("RQ1", "Can late delivery risk be predicted before delivery?", "MEASURED", f"Checkout stage ROC-AUC {A0['test']['roc_auc']:.2f} (weak); carrier hand-off stage {A1['test']['roc_auc']:.2f}. Strong temporal shift in late rate."),
    E("RQ2", "Which sellers/months are statistically unusual?", "MEASURED (no ground truth)", "Anomaly detectors use FDR-controlled tests; accuracy cannot be scored without labels."),
    E("RQ3", "Do richer NLP features improve review understanding?", "PARTLY MEASURED", "Rules/TF-IDF measured on 150 labels from one AI annotator. Multilingual transformer NOT RUN (downloads blocked)."),
    E("RQ4", "Are multi-agent investigations faithful to the data?", "MEASURED (automated)", "Agent vs independent pandas gold on 30 sellers; see Experiment 5."),
    E("RQ5", "Do agents save analyst time?", "NOT RUN", "Needs human participants timed on manual vs agent investigation. Protocol: docs/V2_EXPERIMENT_PROTOCOLS.md"),
    E("RQ6", "Do explanations (SHAP) help users trust/act?", "NOT RUN (user study)", "SHAP computed in v1; human study not possible here."),
    E("RQ7", "Does grounding reduce LLM hallucination?", "NOT RUN", "No local LLM in the build environment. The assistant has a numeric-consistency guard for an optional LLM, tested only with stub functions."),
    E("EXP4", "Explainability (SHAP) evaluation", "v1 MEASURED", "Grouped SHAP importance reported in the v1 report; no human evaluation."),
    E("EXP6", "Recommendation evaluation", "NOT RUN", "Recommendations R1-R7 are evidence-linked in v1; no outcome data exists to evaluate whether they help.")]})
print("status files written")
