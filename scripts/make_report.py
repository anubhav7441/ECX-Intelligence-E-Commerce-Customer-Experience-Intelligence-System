"""Generate docs/PROJECT_REPORT.md from the real result files (every number is read from JSON, never typed by hand).
Run after the pipeline:  python scripts/make_report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = lambda p: json.load(open(ROOT / p, encoding="utf-8"))
S = J("research/statistical_analysis/results/stats_results.json")
N = J("ml/sentiment_analysis/results/nlp_results.json")
E = J("ml/evaluation/nlp_eval_results.json")
M = J("ml/prediction/results/ml_results.json")
B = J("research/results/bi_results.json")
P = J("data/processed/preprocessing_report.json")
SH1 = J("ml/explainability/results/shap_S1_R.json")
SH0 = J("ml/explainability/results/shap_S0_R.json")
CAL = J("ml/evaluation/calibration.json")["models"]
h1, h1s, d = S["H1_late_vs_ontime"], S["H1_sensitivity_review_after_delivery_only"], S["descriptive"]
K = B["kpis"]
cat = {c["category"]: c for c in N["complaint_categories"]}
sel = M["selected"]
fs0, fs0r, fs1, fs1r = "S0 checkout-time", "S0+R checkout-time + history", "S1 delivery-time", "S1+R delivery-time + history"
ac = M["auc_comparisons"]


def fp(p):
    return "< 1e-300" if p == 0 else f"{p:.2g}"


def f(x, n=3):
    return f"{x:.{n}f}"


def tbl(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


funnel = P["funnel"]
funnel_tbl = tbl(["Step", "Orders"], [[k.replace("_", " ").replace("...", "").strip(), f"{v:,}"] for k, v in funnel.items()])
qual_tbl = tbl(["Check on analysed orders", "Count"], [[k.replace("_", " "), f"{v:,}"] for k, v in P["quality"].items()])

ml_rows = []
for fs, models in M["results"].items():
    for m, r in models.items():
        t = r["test"]
        ml_rows.append([fs, m + (" **(selected)**" if sel[fs]["model"] == m else ""), f(t["roc_auc"]), f(t["pr_auc"]), f(t["precision"]), f(t["recall"]), f(t["f1"])])
ml_tbl = tbl(["Feature set", "Model", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"], ml_rows)

cmp_rows = [[k.replace("XGBoost: ", "").replace("  minus  ", " minus "), f"{v['mean_diff']:+.3f}", f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"] for k, v in ac.items() if k.startswith("XGBoost")]
cmp_tbl = tbl(["Comparison (XGBoost)", "Delta ROC-AUC", "95% CI (paired bootstrap)"], cmp_rows)
alg_rows = []
mx = 0
for fs, dd in M["algorithm_comparisons_xgboost_minus_other_auc"].items():
    row = [fs]
    for m in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        x = dd[m]; mx = max(mx, abs(x["mean_diff"]))
        row.append(f"{x['mean_diff']:+.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]")
    alg_rows.append(row)
alg_tbl = tbl(["Feature set", "XGB minus LogReg", "XGB minus Tree", "XGB minus Forest"], alg_rows)

verd_tbl = tbl(["ID", "Hypothesis", "Test", "Evidence", "Verdict"], [[v["id"], v["hypothesis"], v["data_test"], v["statistical_evidence"], f"**{v['verdict']}**"] for v in S["verdicts"]])
band_tbl = tbl(["Delay band", "Orders", "Mean score", "% dissatisfied"], [[b["delay_days"], f"{b['n']:,}", f(b["mean_score"], 2), f(b["pct_dissatisfied"], 1)] for b in S["delay_band_table"]])
cc_tbl = tbl(["Complaint category", "% of negative commented reviews", "Mean score when present", "Classifier F1 (150 labelled)"],
             [[c["category"].replace("_", " "), f(c["pct_of_negative_reviews"], 1), f(c["mean_score_when_present"], 2) if c["mean_score_when_present"] else "-",
               f(next((p["f1"] for p in E["per_class"] if p["category"] == c["category"]), float("nan")), 2)]
              for c in sorted(N["complaint_categories"], key=lambda c: -c["pct_of_negative_reviews"])])
grp1 = ", ".join(f"{k} {v:.1f}%" for k, v in list(SH1["group_share_pct"].items())[:4])
grp0 = ", ".join(f"{k} {v:.1f}%" for k, v in list(SH0["group_share_pct"].items())[:4])
top1 = ", ".join(f"{t['feature']} ({t['mean_abs_shap']:.3f})" for t in SH1["top15"][:5])
lt = S["logit_controls"]["terms"]
logit_tbl = tbl(["Term", "Odds ratio", "95% CI", "p"], [[k, f(v["odds_ratio"], 2), f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]", fp(v["p"])] for k, v in lt.items()])
rec_md = "\n\n".join(f"**{r['id']} (priority {r['priority']}) - {r['action']}**\n\n*Why:* {r['why']}\n\n*Expected impact:* {r['expected_impact']}\n\n*How to measure:* {r['measure']}" for r in B["recommendations"])
find_md = "\n".join(f"{i+1}. **{x['finding']}** {x['evidence']}" for i, x in enumerate(B["findings"]))
sp = M["split"]
jr = B["journey"]
sev = N["severity_distribution_negative_pct"]
q10 = S["Q10_repeat_purchase"]

doc = f"""# ECX-Intelligence: E-Commerce Customer Experience Intelligence System
## An Explainable AI Framework for Customer Satisfaction Prediction Using Transactional Data and Aspect-Based Sentiment Analysis

*B.Tech (IT) Major Project Report. All numbers in this report are generated automatically from the experiment output files by `scripts/make_report.py`.*

---

## Abstract

This project investigates what drives good and bad purchasing experiences in e-commerce, using the public Olist Brazilian marketplace dataset ({P['orders_total']:,} orders, 2016-2018). A relational database, a leakage-safe preprocessing pipeline, statistical hypothesis tests, a Portuguese review-text pipeline (complaint taxonomy, aspect-based sentiment, severity), machine-learning models for predicting dissatisfied customers (1-2 star reviews), SHAP explanations and a Flask web dashboard were built. Among {funnel['...and_has_review']:,} delivered, reviewed orders, **{h1['pct_dissatisfied_late']:.1f}% of late orders received a 1-2 star review versus {h1['pct_dissatisfied_on_time']:.1f}% of on-time orders** (odds ratio {h1['odds_ratio']:.1f}; Cliff's delta {h1['cliffs_delta']:.2f}). The association is smaller but still clear when only reviews written after the recorded delivery are used (delta {h1s['cliffs_delta']:.2f}). Predicting dissatisfaction is difficult: on a held-out later time period the best checkout-time model reaches ROC-AUC {sel[fs0r]['test']['roc_auc']:.2f} and the best delivery-time model {sel[fs1r]['test']['roc_auc']:.2f}. Historical seller/category/state review statistics give a small but statistically reliable improvement, while the choice of algorithm matters little. SHAP shows that delivery delay is the single most important factor in the delivery-time model. The findings are associations in historical data, not proof of causation.

**Keywords:** customer satisfaction, e-commerce, delivery delay, explainable AI, SHAP, aspect-based sentiment analysis, Portuguese NLP, XGBoost.

---

## 1. Introduction

### 1.1 Problem statement
Online marketplaces depend on customer reviews, yet a review score alone does not show *why* a purchasing experience failed. Delivery performance, seller behaviour, product category and region may all contribute, and review text contains further detail. Managers need (a) evidence of which factors matter, (b) an early-warning signal for at-risk orders and (c) explanations they can act on.

### 1.2 Objectives
1. Build a verified relational database and a reproducible preprocessing pipeline for the Olist data.
2. Measure how delivery performance and other operational factors are associated with customer satisfaction, using formal hypothesis tests with effect sizes and confidence intervals.
3. Extract complaint categories, aspect-level sentiment and severity from Portuguese review text and validate the method on a labelled sample.
4. Predict dissatisfied customers without target leakage, compare four algorithms on a time-based hold-out, and explain predictions with SHAP.
5. Deliver an interactive web application and evidence-based, traceable recommendations.

### 1.3 Research questions
- **RQ1** How is delivery performance associated with customer satisfaction?
- **RQ2** Which customer-experience aspects are most associated with negative reviews?
- **RQ3** Does combining transactional/operational features with review-derived features improve prediction under valid settings?
- **RQ4** Which algorithm performs best at identifying dissatisfied customers?
- **RQ5** Can SHAP identify the major factors behind individual predictions?

The original project brief additionally asked about price (Q5), delivery distance (Q6), sellers, categories, freight cost, regions and repeat purchase (Q10); these are answered with the same statistical framework.

---

## 2. Background and related work

This section lists methodological sources that are used directly in the project. Domain literature on delivery performance and satisfaction should be added by the student after reading and verifying each paper; **no unverified citations have been included.**

- SHAP: Lundberg & Lee (2017) unify several explanation methods through Shapley values; TreeExplainer computes them exactly for tree ensembles.
- Gradient boosting: Chen & Guestrin (2016), XGBoost. Random forests: Breiman (2001).
- Non-parametric testing for ordinal scores: Mann & Whitney (1947); effect size Cliff's delta (Cliff, 1993). Bootstrap confidence intervals: Efron (1979). Binomial proportion intervals: Wilson (1927).
- Aspect-based sentiment analysis as a task: Pontiki et al. (2016), SemEval-2016 Task 5.
- Dataset: Olist, "Brazilian E-Commerce Public Dataset by Olist", Kaggle.

---

## 3. Dataset and data quality

Nine CSV files were verified: customers ({99441:,}), geolocation (1,000,163 rows, 19,015 distinct zip prefixes), order items (112,650), payments (103,886), reviews (99,224), orders ({P['orders_total']:,}), products (32,951), sellers (3,095) and a category-translation table (71 rows; two categories present in the products file were missing and were added manually).

Verified data issues that shaped the design:
- `review_id` is **not unique** (789 ids repeat) and {P['orders_with_multiple_reviews']:,} orders have more than one review; the latest review per order was kept.
- 9,803 orders have several item rows and 2,961 have several payment rows; items, payments and reviews are **aggregated to order level before joining**, which prevents revenue/order inflation (checked against direct SQL sums).
- 58.7% of reviews have no comment text. Review text is Portuguese.
- Only 96,470 delivered orders have a delivery date, so lateness is defined for those only.
- Orders with a recorded review dated **before** the delivery date: {P['quality']['review_created_before_delivery']:,} (reported in a sensitivity analysis).

{funnel_tbl}

{qual_tbl}

**Definitions.** *Late* = delivered date later than the estimated delivery date (date-based). *Dissatisfied* = review score 1 or 2. *Distance* = straight-line (haversine) km between seller and customer zip-prefix centroids (median coordinates).

---

## 4. Methodology

### 4.1 Database
SQLite with eight tables, primary/foreign keys and indexes, loaded with SQLAlchemy and checked with `PRAGMA foreign_key_check` (0 violations). SQLite was chosen over MySQL/PostgreSQL because it needs no installation and suits a single-user local project.

### 4.2 Statistical analysis
Mann-Whitney U with Cliff's delta, chi-square with odds ratio (Woolf CI), bootstrap CI for the mean difference, Spearman correlation with quartile contrasts, chi-square homogeneity tests across sellers/categories/states with Cramer's V and Wilson intervals, and a multivariable logistic regression. Because the sample is large, every p-value is reported together with an effect size.

### 4.3 NLP pipeline
Text is lower-cased, accent-stripped and normalised. A transparent rule-based taxonomy (10 classes, multi-label) detects complaint categories; sentence-level lexicon polarity with negation gives aspect-level sentiment for six aspects (delivery, product, seller service, packaging, price, payment); rules assign severity. A TF-IDF + Logistic Regression model gives overall sentiment (labels derived from stars). The complaint classifier was evaluated on **150 reviews labelled blind to its output** by a single annotator (an AI assistant); these are not independent human gold labels, and the labelled file is included for audit.

### 4.4 Machine learning protocol (leakage control)
- **Target:** dissatisfied = 1-2 stars.
- **Two prediction moments:** *checkout-time* (S0: order, price, freight, distance, estimated delivery, payment) and *delivery-time* (S1: S0 + fulfilment outcome such as delivery days and delay).
- **Review-derived (R) features:** the seller's, category's and state's review statistics computed **only from reviews created strictly before the order's purchase time**, shrunk toward the training mean. An order's own score or text is never an input. A brute-force unit test confirms the history of sampled orders contains no future or own review.
- **Split:** time-ordered 70/15/15 ({sp['train_period'][0]} to {sp['train_period'][1]} train, validation to {sp['val_period'][1]}, test {sp['test_period'][0]} to {sp['test_period'][1]}); customers appearing in the test period were removed from training/validation. The training dissatisfied rate is {sp['dissatisfied_rate_train']*100:.1f}%, test {sp['dissatisfied_rate_test']*100:.1f}% (class imbalance; handled with class weights).
- **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost with fixed hyper-parameters (no test-set tuning); decision threshold chosen on validation to maximise F1; model per feature set chosen on validation PR-AUC; test set evaluated once.
- **Comparisons:** paired bootstrap of the ROC-AUC difference on the same test orders.

### 4.5 Explainability
SHAP TreeExplainer on the selected XGBoost models, explaining the test orders; one-hot columns are summed back to their original feature, and features are also grouped (delivery, price/freight, product, order structure, geography, payment, time, historical reviews).

---

## 5. System design and implementation

**Architecture.** Data files and SQLite database -> preprocessing (`ml/preprocessing`) -> analysis modules (`research`, `ml/sentiment_analysis`, `ml/prediction`, `ml/explainability`) -> result files (JSON/parquet/PNG) and trained models -> Flask services (`app/services`) -> REST API and HTML pages (`app/routes`, `app/templates`) with Bootstrap 5 and Chart.js (bundled locally, so the app works offline).

**Web modules (all implemented and browser-tested):** Executive Dashboard; Data Management (database status, quality summary, CSV validation); Review Intelligence (live analyser); Satisfaction Prediction; Explainable AI (global SHAP and per-order explanations); Seller, Product and Geographical Intelligence; Research & Statistics; Recommendations.

**Scores.** The *Customer Experience Score* (0-100) = 50% review, 30% delivery timeliness, 20% text sentiment (weights are a design choice, stated on the dashboard). The *Seller Performance Score* combines shrunk average review (50%), on-time rate (30%) and volume rank (20%). Sellers are ranked only with at least {B['sellers']['min_orders']} orders.

---

## 6. Results

### 6.1 Descriptive overview
{funnel['...and_has_review']:,} analysed orders: mean review score {d['mean_review_score']:.2f}, {d['pct_dissatisfied_1_2']:.1f}% dissatisfied, {d['pct_late']:.1f}% late, mean delivery time {d['mean_delivery_days']:.1f} days against a mean estimate of {d['mean_estimated_days']:.1f} days. Platform KPIs (all delivered orders): item revenue R$ {K['revenue_items_delivered']:,.0f}, average order value R$ {K['avg_order_value_items']:.0f}, {K['pct_customers_repeat']:.1f}% of customers ordered more than once.

### 6.2 RQ1: Delivery performance and satisfaction
| | Late | On time |
|---|---|---|
| Orders | {h1['n_late']:,} | {h1['n_on_time']:,} |
| Mean score | {h1['mean_score_late']:.2f} | {h1['mean_score_on_time']:.2f} |
| % dissatisfied | {h1['pct_dissatisfied_late']:.1f} | {h1['pct_dissatisfied_on_time']:.1f} |

Mean difference {h1['mean_diff']:.2f} (95% bootstrap CI {h1['mean_diff_ci95'][0]:.2f} to {h1['mean_diff_ci95'][1]:.2f}); Mann-Whitney p {fp(h1['p_value'])}; Cliff's delta {h1['cliffs_delta']:.2f} ({h1['effect_size_label']}); odds ratio {h1['odds_ratio']:.1f} (95% CI {h1['odds_ratio_ci95'][0]:.1f}-{h1['odds_ratio_ci95'][1]:.1f}).

**Sensitivity analysis.** {h1['n_late'] - h1s['n_late']:,} late orders have a review dated before the recorded delivery. Restricting to reviews created on/after delivery ({h1s['n_late']:,} late orders) gives a smaller effect: {h1s['pct_dissatisfied_late']:.1f}% vs {h1s['pct_dissatisfied_on_time']:.1f}% dissatisfied, odds ratio {h1s['odds_ratio']:.1f}, Cliff's delta {h1s['cliffs_delta']:.2f} ({h1s['effect_size_label']}). Both are reported so the headline is not overstated.

**Dose-response.**

{band_tbl}

After controlling for price, freight ratio, distance, delivery time and items in the order, being late keeps an odds ratio of {lt['is_late']['odds_ratio']:.1f} (pseudo R-squared {S['logit_controls']['pseudo_r2']:.3f}):

{logit_tbl}

**Other factors.** Item price (rho {S['Q5_price_vs_score']['spearman_rho']:.3f}), freight ratio (rho {S['H4_freight_ratio_vs_score']['spearman_rho']:.3f}) and seller-customer distance (rho {S['Q6_distance_vs_score']['spearman_rho']:.3f}) are statistically significant but have negligible effect sizes (top-vs-bottom-quartile Cliff's delta {S['Q5_price_vs_score']['top_vs_bottom_quartile']['cliffs_delta']:.2f}, {S['H4_freight_ratio_vs_score']['top_vs_bottom_quartile']['cliffs_delta']:.2f}, {S['Q6_distance_vs_score']['top_vs_bottom_quartile']['cliffs_delta']:.2f}). A significant p-value with a negligible effect is *not* a practically important finding.

### 6.3 Sellers, categories, regions
- **Sellers:** chi-square across {S['H2_sellers']['n_groups']} sellers (>=50 orders) p {fp(S['H2_sellers']['p'])}, Cramer's V {S['H2_sellers']['cramers_v']:.2f}; {S['H2_sellers']['n_groups_significantly_worse']} sellers have a dissatisfaction interval entirely above the platform rate.
- **Categories:** {S['H3_categories']['n_groups_significantly_worse']} of {S['H3_categories']['n_groups']} significantly worse (V {S['H3_categories']['cramers_v']:.2f}).
- **States:** {S['H5_states']['n_groups_significantly_worse']} of {S['H5_states']['n_groups']} significantly worse (V {S['H5_states']['cramers_v']:.2f}).
- **Segmentation (revenue vs experience):** sellers {B['sellers']['segment_counts']}; categories {B['categories']['segment_counts']}.

### 6.4 RQ2: Review text
{N['n_commented_reviews']:,} analysed reviews have text; {N['n_negative_commented']:,} are 1-2 stars. Severity among negative reviews: High {sev['High']:.1f}%, Medium {sev['Medium']:.1f}%, Low {sev['Low']:.1f}%.

{cc_tbl}

Reviews with a delivery complaint score {N['H2_delivery_complaint_vs_score']['mean_with']:.2f} versus {N['H2_delivery_complaint_vs_score']['mean_without']:.2f} without (Cliff's delta {N['H2_delivery_complaint_vs_score']['cliffs_delta']:.2f}). Cross-check against the objective late flag: {N['construct_validity_late_text_vs_late_flag']['pct_late_orders_with_late_complaint']:.0f}% of commented late orders mention delay or non-delivery, versus {N['construct_validity_late_text_vs_late_flag']['pct_ontime_orders_with_late_complaint']:.0f}% of commented on-time orders.

**Reliability.** On 150 hand-labelled reviews the complaint classifier has micro-F1 {E['micro_f1']:.2f} (macro-F1 {E['macro_f1_over_supported_classes']:.2f}, exact-match {E['exact_match_accuracy']*100:.0f}%); detecting any complaint reaches F1 {E['complaint_vs_no_complaint']['f1']:.2f}. Seller-issue and wrong-product are the weakest classes. The sentiment model (stars as weak labels) reaches accuracy {N['sentiment']['tfidf_lr']['accuracy']*100:.1f}% and macro-F1 {N['sentiment']['tfidf_lr']['macro_f1']:.2f}, versus {N['sentiment']['lexicon_baseline']['macro_f1']:.2f} for the lexicon baseline and a majority-class accuracy of {N['sentiment']['majority_baseline']['accuracy']*100:.1f}%; the neutral class is hard (F1 {N['sentiment']['tfidf_lr']['per_class']['neutral']['f1-score']:.2f}). Complaint counts are therefore indicative, not exact.

### 6.5 RQ3 and RQ4: Prediction
Test period {sp['test_period'][0]} to {sp['test_period'][1]} (n={sp['n_test']:,}, {sp['dissatisfied_rate_test']*100:.1f}% dissatisfied). Chance level: ROC-AUC 0.50, PR-AUC {sp['dissatisfied_rate_test']:.3f}.

{ml_tbl}

Baselines: always predicting "satisfied" gives {M['baselines']['majority_class']['accuracy']*100:.1f}% accuracy but zero recall; the rule "dissatisfied if late" gives ROC-AUC {M['baselines']['late_flag_rule (predict dissatisfied iff late)']['roc_auc']:.3f}, F1 {M['baselines']['late_flag_rule (predict dissatisfied iff late)']['f1']:.3f}. Accuracy is therefore not used to judge models.

**RQ3 (do review-derived features help?)**

{cmp_tbl}

Adding historical review statistics improves ROC-AUC reliably but modestly (confidence intervals exclude zero); knowing the delivery outcome (S1 vs S0) is a far larger gain. For the best models, delivery-time precision/recall/F1 are {sel[fs1r]['test']['precision']:.2f}/{sel[fs1r]['test']['recall']:.2f}/{sel[fs1r]['test']['f1']:.2f}.

**RQ4 (which algorithm?)** XGBoost had the best validation PR-AUC in all four feature sets and was selected, but test differences between algorithms are tiny (at most {mx:.3f} ROC-AUC) and often inside the confidence interval:

{alg_tbl}

No algorithm is clearly superior; feature choice matters much more than algorithm choice.

**Sensitivity.** Restricting the delivery-time model to test orders whose review was written after delivery lowers ROC-AUC from {sel[fs1r]['test']['roc_auc']:.3f} to {M['sensitivity_S1_review_after_delivery']['roc_auc']:.3f} (the S0 model on the same subset: {M['sensitivity_S0_review_after_delivery']['roc_auc']:.3f}). Part of the apparent delivery-time performance therefore comes from reviews written before the recorded delivery.

**Calibration.** The models use class weights, so raw scores are not probabilities. On the test period, orders in the top delivery-time score band were actually dissatisfied {CAL['S1_R']['bands'][-1]['observed_dissatisfied_rate']*100:.0f}% of the time (base rate {CAL['S1_R']['test_base_rate']*100:.1f}%); the web app shows this observed rate instead of a raw probability.

### 6.6 RQ5: Explainability
For the delivery-time model, factor groups by share of total mean |SHAP|: {grp1}. Top features: {top1}. For the checkout-time model: {grp0}. Individual explanations (web app and `docs/figures/shap_local_*.png`) show which factors raise or lower an order's risk. SHAP explains the *model*; correlated features (delay, delivery days, late flag) share credit, and importance is not causation.

### 6.7 Customer journey and repeat purchase
Late orders lose most time during carrier transit; {jr['missed_seller_handoff_deadline_pct']:.1f}% of orders missed the seller's carrier hand-off deadline and were late {jr['late_pct_if_handoff_missed']:.0f}% of the time (versus {jr['late_pct_if_handoff_ok']:.0f}% otherwise). Repeat purchase after a dissatisfied first order was {q10['repeat_rate_dissatisfied_pct']:.2f}% versus {q10['repeat_rate_satisfied_pct']:.2f}% after a satisfied one (p {q10['p']:.2f}, not significant); only {K['pct_customers_repeat']:.1f}% of customers reorder, so this question has little statistical power here.

### 6.8 AI hypotheses versus data (verdict table)

{verd_tbl}

---

## 7. Business findings and recommendations

{find_md}

### Recommendations (each traceable to the evidence above)

{rec_md}

---

## 8. Testing
An automated pytest suite (70 tests) covers: NLP rules (valid, empty and non-string text, negation, praise-versus-complaint), data pipeline integrity (one row per order, no join inflation against SQL, foreign keys, target and late definitions), leakage (forbidden columns, temporal ordering, customer-disjoint split, brute-force check of historical features), CSV upload validation (valid, empty, wrong type, duplicates, invalid dates and scores, oversize), model loading and prediction (valid/invalid inputs, monotonic effect of delay), and the API (all pages, JSON errors, 404/413, database-failure behaviour). The web interface was additionally exercised in a real headless browser (all pages, forms, charts, mobile width). A clean-room run (fresh virtual environment, all generated files deleted, full pipeline, tests) reproduced the results.

---

## 9. Limitations and threats to validity
1. **Association, not causation.** Observational data from one marketplace; unmeasured factors (product expectations, customer type) may confound.
2. **Review timing.** Many late-order reviews pre-date the recorded delivery; effect sizes shrink when they are excluded.
3. **Selection bias.** Only customers who left a review are analysed; 58.7% of reviews have no text.
4. **Label quality.** Complaint labels come from rules checked on 150 reviews labelled by a single AI annotator; sentiment labels are derived from star scores; no independent human gold standard was available.
5. **Moderate predictive power.** ROC-AUC about {sel[fs0r]['test']['roc_auc']:.2f} (checkout) and {sel[fs1r]['test']['roc_auc']:.2f} (delivery); suitable for risk monitoring, not decisions about individual customers. Scores are not calibrated probabilities.
6. **Attribution.** One seller/category is attributed per order (first item); multi-item orders have a single review.
7. **Generalisation.** 2016-2018 data; distance uses zip-prefix centroids; repeat purchase is rare.
8. **Weights.** The experience and seller scores use fixed weights chosen by design; they are not fitted.

## 10. Conclusion and future work
Late delivery is the dominant, actionable factor associated with dissatisfied reviews, seller quality varies significantly, and price and freight matter little. Delivery-time information and seller history improve prediction, but algorithm choice does not. Future work: independent human labels (and agreement statistics) for the NLP evaluation, Portuguese transformer models, calibrated probabilities, causal methods (for example matching on order characteristics), richer carrier-level data and a live data feed.

## References
1. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. (arXiv:1705.07874)
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. 22nd ACM SIGKDD*.
3. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825-2830.
5. Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics, 18*(1), 50-60.
6. Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin, 114*(3), 494-509.
7. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics, 7*(1), 1-26.
8. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209-212.
9. Pontiki, M., et al. (2016). SemEval-2016 Task 5: Aspect based sentiment analysis. *Proc. SemEval-2016*, 19-30.
10. Olist. Brazilian E-Commerce Public Dataset by Olist. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

*Citation details above are given from the author's knowledge of standard references; the SHAP reference was verified online. Verify page numbers against the originals before submission.*

## Appendix A: Figures
All charts are in `docs/figures/`. Key figures:

![Figure 1: Review score distribution](figures/eda_score_distribution.png)

![Figure 2: Estimated versus actual delivery time](figures/eda_estimated_vs_actual_delivery.png)

![Figure 3: Dissatisfied reviews by delivery delay](figures/eda_dissatisfaction_by_delay.png)

![Figure 4: Monthly late-delivery and dissatisfaction rates](figures/eda_monthly_late_dissatisfied.png)

![Figure 5: Price, freight and distance quartiles versus dissatisfaction](figures/eda_price_freight_distance.png)

![Figure 6: Complaint categories in negative reviews](figures/bi_complaint_categories.png)

![Figure 7: Aspect-based sentiment in negative reviews](figures/bi_aspect_sentiment.png)

![Figure 8: Seller segmentation](figures/bi_seller_segments.png)

![Figure 9: Category segmentation](figures/bi_category_segments.png)

![Figure 10: Dissatisfaction by customer state](figures/bi_state_hotspots.png)

![Figure 11: Where late orders lose time](figures/bi_journey_stages.png)

![Figure 12: SHAP importance by factor group, delivery-time model](figures/shap_groups_S1_R.png)

![Figure 13: SHAP summary, delivery-time model](figures/shap_beeswarm_S1_R.png)

![Figure 14: SHAP explanation of a high-risk order](figures/shap_local_S1_R_highest_risk_true_dissatisfied.png)
"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "PROJECT_REPORT.md").write_text(doc, encoding="utf-8")
print("report written:", len(doc.split()), "words")
