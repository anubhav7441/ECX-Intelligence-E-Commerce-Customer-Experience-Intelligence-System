"""Run the COMPLETE analysis pipeline in the correct order.   python scripts/run_all.py
Prerequisite: the 9 Olist CSV files are in data/raw/ (see README). Takes roughly 5-8 minutes on a normal laptop."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("1/9 Load CSVs into the SQLite database", ["scripts/01_load_database.py"]),
    ("2/9 Build order-level analytic dataset", ["-m", "ml.preprocessing.build_dataset"]),
    ("3/9 Statistical analysis (hypothesis tests)", ["-m", "research.statistical_analysis.run_stats"]),
    ("4/9 Review NLP + sentiment model", ["-m", "ml.sentiment_analysis.run_review_pipeline"]),
    ("5/9 Evaluate NLP on hand-labelled sample", ["-m", "ml.evaluation.evaluate_nlp"]),
    ("6/9 Train and compare ML models", ["-m", "ml.prediction.train_models"]),
    ("7/9 Score calibration on the test period", ["-m", "ml.evaluation.calibration"]),
    ("8/9 SHAP explainability", ["-m", "ml.explainability.run_shap"]),
    ("9/9 EDA charts, BI tables, recommendations", ["-m", "research.run_eda_bi"]),
]


def main():
    raw = ROOT / "data" / "raw"
    need = ["olist_customers_dataset.csv", "olist_geolocation_dataset.csv", "olist_order_items_dataset.csv",
            "olist_order_payments_dataset.csv", "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
            "olist_products_dataset.csv", "olist_sellers_dataset.csv", "product_category_name_translation.csv"]
    missing = [f for f in need if not (raw / f).exists()]
    if missing:
        sys.exit("Missing files in data/raw/: " + ", ".join(missing) + "\nDownload the Olist dataset from Kaggle first (see README).")
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for label, args in STEPS:
        print(f"\n=== {label} ===", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, *args], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"\nSTEP FAILED: {label} (exit code {r.returncode}). Fix the error above and run again.")
        print(f"--- done in {time.time() - t:.0f}s", flush=True)
    print(f"\nALL STEPS COMPLETED in {time.time() - t0:.0f}s. Start the app with:  python run.py")


if __name__ == "__main__":
    main()
