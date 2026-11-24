"""Writes docs/RECOMMENDATION_SYSTEM.md; every number is read from recsys/results/evaluation.json and prep_log.json."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
e = json.loads((ROOT / "recsys/results/evaluation.json").read_text()); log = json.loads((ROOT / "data/processed/recsys/prep_log.json").read_text())["decisions"]
t = e["protocol"]["test"]; v = e["protocol"]["validation"]; R = e["test_results"]
names = {"popularity": "Popularity baseline", "item_cf": "Item-based CF", "user_cf": "User-based CF", "als": "Implicit ALS", "content": "Content-based", "assoc": "Association rules only", "hybrid": "Hybrid (validation-tuned)"}
rows = "\n".join(f"| {names[k]} | {x['P@10']:.4f} | {x['R@10']:.4f} | {x['MAP@10']:.4f} | {x['NDCG@10']:.4f} | [{x['NDCG@10_ci95'][0]:.4f}, {x['NDCG@10_ci95'][1]:.4f}] | {x['coverage@10']*100:.2f}% | {x['novelty@10']:.1f} | {x['signal_rate']*100:.0f}% |" for k, x in R.items())
d = e["paired_differences"]; dn = d["hybrid_minus_popularity_NDCG@10"]; dr = d["hybrid_minus_popularity_R@10"]
best = e["hybrid_weights_chosen_on_validation"]; val5 = "\n".join(f"| {i+1} | {x['ndcg10']:.4f} | {x['recall10']:.4f} | {json.dumps({k: w for k, w in x['weights'].items() if w})} |" for i, x in enumerate(e["validation_top5"]))
doc = """# Recommendation Intelligence Center (ECX 2.0 add-on)

Status labels: **MEASURED** = produced by `python -m recsys.evaluate` and stored in `recsys/results/evaluation.json`; **TESTED** = covered by `tests/test_recsys.py`; **PROPOSED** = not implemented.

## 1. What was built
A hybrid product recommender on the Olist data, using real purchase behaviour only. It is a separate package (`recsys/`), a blueprint (`app/routes/recsys.py`), one page (`/recsys`) and 10 JSON endpoints (`/api/rec/*`). The existing database, Olist dashboard and trained satisfaction models were not changed. No new libraries were added (numpy, scipy, scikit-learn, pandas only; ALS and association rules are implemented in the project).

## 2. Architecture
`raw CSVs -> recsys/data.py -> interactions / baskets / products (parquet) -> recsys/models.py (6 components) -> recsys/evaluate.py (validation weight search, one test run) -> recsys/service.py (hybrid, cold start, explanations) -> Flask API -> /recsys page`.

## 3. Dataset preparation (MEASURED, from `prep_log.json`)
""" + "\n".join(f"- {l}" for l in log) + f"""

Important consequence: most customers bought one product once. Co-purchase learning (CF, ALS, rules) can only use the small group of customers with 2+ products or orders with 2+ products.

## 4. Algorithms
- **Popularity (baseline):** score = 0.5 x (buyers / max buyers) + 0.5 x (buyers in the last 90 days before the cutoff / max). Category popularity is used for category-filtered new-customer lists.
- **Item-based CF:** binary matrix X (customers x products) restricted to customers with 2+ distinct products. Similarity of items i, j = co(i,j) / (sqrt(n_i n_j) + 1) (cosine with shrinkage 1 so pairs with tiny support are damped); top-50 neighbours kept. Score of candidate j = sum of sim(h, j) over the customer's purchased items h.
- **User-based CF:** cosine between the customer's items and each 2+-product customer (shrinkage 1), top-50 neighbours, score = similarity-weighted neighbour purchases.
- **Implicit ALS:** confidence c = 1 + 20 x r on binary r, 16 factors, L2 0.1, 15 alternating least-squares iterations (own numpy implementation), trained on 2+-product customers; any customer is scored by solving for their factor vector from their items ("fold-in").
- **Content-based:** TF-IDF of the category name (the data has no product names or descriptions) plus standardised log weight, log volume, log median price, photo count and description length, with missing-value flags (never invented); L2-normalised; cosine similarity; the customer profile is the mean of purchased product vectors.
- **Association rules:** pair counts over real orders with 2+ distinct products. support = pair/orders, confidence = pair/count(A), lift = confidence / (count(B)/orders). Product rules need >= 2 orders; category rules need >= 5. Computed directly (pairs only), not with Apriori, because baskets are small; a unit test recomputes a rule from the raw basket table.
- **Hybrid:** each model's scores are divided by that customer's maximum, then combined as a weighted sum. Weights were NOT chosen by hand: all 70 combinations of the five components in steps of 0.25 were scored on a validation period and the best NDCG@10 was kept. Scores rank products; they are not purchase probabilities.

## 5. Evaluation method (MEASURED)
- Chronological split, no future information in training. Validation: train before {v['train_before']}, evaluate {v['period']} ({v['eligible_customers']} customers). **Test (run once): train before {t['train_before']}, evaluate {t['period']}.**
- Eligible test customers: {t['eligible_customers']} (had a training purchase and bought a product in the period that they had not bought before, and which existed in the training catalog). Held-out purchases: {t['truth_items']}. Training customers: {t['train_users']:,}. Catalog: {t['catalog_size']:,} products.
- Ground truth excludes repurchases and products that did not exist in training (a model cannot recommend unseen products).
- Metrics at K=10 (also K=5, 20 in the JSON): Precision, Recall, MAP, NDCG, catalog coverage (share of the catalog recommended at least once), novelty (mean -log2 of product popularity); 95% bootstrap ranges (1000 resamples of customers).

## 6. Results (MEASURED, test period)
| Method | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | NDCG@10 95% range | Coverage@10 | Novelty | Customers with a real signal |
|---|---|---|---|---|---|---|---|---|
{rows}

Validation top 5 weight combinations (hybrid was built from the first row):

| Rank | NDCG@10 | Recall@10 | Weights |
|---|---|---|---|
{val5}

Hybrid minus popularity, NDCG@10: {dn[0]:+.4f} (95% range [{dn[1]:+.4f}, {dn[2]:+.4f}]); Recall@10: {dr[0]:+.4f} (95% range [{dr[1]:+.4f}, {dr[2]:+.4f}]).

**Reading the results honestly.** All methods score very low in absolute terms and the ranges are wide, because only {t['eligible_customers']} customers could be evaluated and each has about one relevant product among {t['catalog_size']:,}. The popularity baseline is not beaten by the hybrid in this test; the differences between methods are inside the bootstrap ranges, so no method can be declared better. The weights picked on validation (a noisy sample of {v['eligible_customers']} customers) did not carry over. Content-based and ALS reach far more of the catalog (coverage, novelty) at similar accuracy, which is a diversity benefit, not an accuracy benefit. Do not present these as a working conversion engine.

## 7. Cold start (TESTED)
- **New customer (no history):** popular products (optionally within a chosen category), labelled "Popularity fallback". Nothing about preferences is invented.
- **One purchase:** personalised signals are used where they exist; slots with no signal are filled by popularity and labelled as fallback.
- **New / thin products:** under 3 buyers, similarity relies on product details only and a warning is shown; products without a category use size, weight and price only.
- **No product-level pattern:** "bought together" falls back to category-level patterns and says so.

## 8. Explanations (TESTED)
Each recommendation lists reasons built from the signals that scored it: the exact rule (orders containing both, confidence, lift), the nearest purchased item and its similarity, a latent-factor note for ALS, the content similarity, or the popularity counts. SHAP is not used; it does not suit these models.

## 9. Limitations
Sparse data (see section 3); small test set; no product names or text; only 2016-2018 and one marketplace; offline metrics only (no live test, so the effect on real sales is unknown); the association lift for rare pairs is inflated (read with the order count); scores are rankings, not probabilities.

## 10. PROPOSED (not done)
Per-segment evaluation with more data (for example another dataset with more repeat buyers); a text-based content model if product descriptions become available; tuning on a rolling set of validation windows; an online A/B test.

## 11. Reproduce
`python -m recsys.data` (needs the raw CSVs) -> `python -m recsys.evaluate` (about 2 minutes) -> `python -m recsys.make_doc`. The app only needs the files in `data/processed/recsys/` and `recsys/results/`, which are included in the ZIP.
"""
(ROOT / "docs/RECOMMENDATION_SYSTEM.md").write_text(doc); print(len(doc))
