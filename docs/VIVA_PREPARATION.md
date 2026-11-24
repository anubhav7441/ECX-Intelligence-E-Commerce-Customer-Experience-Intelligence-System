# Viva Preparation - ECX-Intelligence

## 1. 60-second explanation
"Online shoppers leave reviews, but a star rating does not say *why* the experience was bad. I used the Olist Brazilian marketplace data (99,441 orders) to find what drives dissatisfied customers. I built a database, cleaned and joined the data carefully, tested hypotheses statistically, analysed the Portuguese review text for complaint types, trained models to predict 1-2 star reviews without leaking the answer, and explained them with SHAP. The strongest finding: 62% of late orders get 1-2 stars versus 9% of on-time orders. Everything is shown in a Flask web app. I also report limits honestly: it is observational data, so these are associations, and prediction is only moderately accurate."

## 2. Key numbers to remember (all from the results files)
- Orders analysed: 95,824 (delivered, with delivery date, with a review). Dissatisfied (1-2 stars): 12.8%. Late: 6.7%.
- Late vs on-time dissatisfied: 62.4% vs 9.3%; odds ratio 16.3; Cliff's delta -0.64 (large).
- Sensitivity (reviews written after delivery only): 25.1% vs 9.2%; delta -0.30 (small).
- Test split: 2018-06-21 to 2018-08-29, n=14,374, 9.6% dissatisfied.
- Best models (XGBoost) test ROC-AUC: checkout-time 0.61, delivery-time 0.71.
- NLP complaint classifier on 150 hand-labelled reviews: micro-F1 0.65. Sentiment model accuracy 80.2% (majority class 67.4%).
- 70 automated tests; clean rebuild reproduces all results exactly.

## 3. Technologies - one-line explanations
- **Python/pandas/NumPy:** data handling and calculations. **SQLite + SQLAlchemy + SQL:** relational storage; SQLAlchemy lets Python talk to the database. **SciPy/statsmodels:** hypothesis tests, regression.
- **scikit-learn:** pipelines, Logistic Regression, Decision Tree, Random Forest, TF-IDF, metrics. **XGBoost:** boosted trees. **SHAP:** explains how each feature moves a prediction.
- **Flask:** Python web framework; routes return HTML pages or JSON (REST API). **Bootstrap/Chart.js:** layout and charts in the browser. **pytest:** automated tests.

## 4. Likely examiner questions and answers
**Why this dataset?** Public, real, multi-table, has delivery dates and reviews, so operations and satisfaction can be linked.
**Why SQLite, not MySQL?** No installation or server, enough for ~100k orders, SQL is the same. Switching only needs a different connection string in `config.py`.
**What is a primary / foreign key here?** `orders.order_id` is the primary key; `order_items.order_id` is a foreign key to it. `order_items` uses a composite key (order_id, order_item_id). Reviews needed a surrogate key because `review_id` is not unique (789 repeated ids).
**What is a join inflation problem?** An order with 3 items joined to 2 payments creates 6 rows, so revenue is counted 6 times over. I aggregate items, payments and reviews to one row per order first, and a test compares my revenue with a direct SQL SUM.
**How did you define late?** Delivered date later than estimated date, at date level. Only for delivered orders with a delivery date (96,470).
**What is target leakage and how did you avoid it?** Using information unavailable at prediction time or derived from the answer. The order's own score/text is never an input; history features use only reviews created before the order's purchase time (brute-force test); the split is by time and customer-disjoint; two prediction moments (checkout vs delivery) are kept separate.
**Why not just report accuracy?** Only ~10-13% are dissatisfied; predicting "satisfied" for everyone gives ~90% accuracy and finds nobody. I use ROC-AUC, PR-AUC, precision, recall, F1 and confusion matrices.
**Why time-based split?** Real use predicts the future; random splits leak patterns across time. Test period is later than training.
**Your ROC-AUC is only 0.57-0.71. Is that bad?** It is modest and I say so. Satisfaction depends on things not in the data. Delivery information raises ROC-AUC by about 0.12 over checkout-only. The model is useful for risk monitoring, not individual decisions.
**Which algorithm is best?** No clear winner: differences are within about 0.014 ROC-AUC. I selected XGBoost on validation PR-AUC, but the paired bootstrap shows they are nearly equal. Features matter more than algorithm.
**What is SHAP?** A method based on Shapley values from game theory: it splits a prediction among features. Positive values push toward "dissatisfied". TreeExplainer is exact for tree models. It explains the model, not real-world causes.
**What are the SHAP limitations?** Correlated features share credit (delay, delivery days, late flag); values are in log-odds; no causation.
**Is "late delivery causes bad reviews" proven?** No. It is an association from observational data. Controlling for price, freight, distance, delivery time and item count, being late still has an odds ratio of 7.7, but unmeasured factors could play a role. Also many late-order reviews are dated before the recorded delivery, which is why I report a sensitivity analysis.
**Why are p-values like 1e-300 reported with effect sizes?** With ~96,000 orders even tiny differences are "significant". Effect size (Cliff's delta, Cramer's V, odds ratio) says whether it matters. Price/freight/distance: significant but negligible.
**What is Cliff's delta?** Probability that a random late-order score is lower than an on-time score minus the reverse; -1 to 1; |0.474| or more is large.
**How did you handle Portuguese text?** Accent-stripping and normalisation, then Portuguese keyword rules and negation handling; TF-IDF with Logistic Regression for sentiment. English-only tools (e.g. NLTK VADER) do not work on Portuguese.
**How accurate is the complaint classifier?** Micro-F1 0.65 on 150 reviews labelled by one annotator (an AI assistant, blind to the classifier output). Not an independent human gold standard; I state this limitation, and the labelled CSV is included for re-labelling.
**Sentiment labels come from stars - is that circular?** It is a weak-label proxy, stated as such: it measures whether text predicts the star class, not human sentiment.
**What is class imbalance and how did you handle it?** Few positive cases. I used class weights/`scale_pos_weight`, chose the threshold on validation, and used PR-AUC.
**Why are model scores not probabilities?** Class weights distort them. The app shows the observed dissatisfied rate among test orders with a similar score instead.
**What is Wilson interval / why minimum sample sizes?** A confidence interval for a proportion that behaves well for small samples; a seller with 3 orders can look perfect or terrible by chance, so only sellers with at least 50 orders are ranked.
**Biggest limitations?** Observational single-marketplace 2016-2018 data; only reviewers observed; label quality; modest prediction; one seller/category per order.
**What would you do next?** Independent human labels and agreement statistics, transformer models for Portuguese, calibrated probabilities, causal methods, live data.
**What did you change from the original project idea?** Power BI replaced by a Flask dashboard; LLM replaced by local NLP; added ML, SHAP and a web app; kept and restored the AI-vs-data verdicts, customer journey, price/distance/repeat questions and recommendation format.
**Did you copy any code/library output without checking?** Every number is produced by code in the repository and tests confirm key properties; a clean rebuild reproduces all results.

## 5. Possible tricky challenges
- "Your late-delivery effect shrinks after excluding early reviews." Yes: delta -0.64 to -0.30; both reported, the effect is still clear and statistically strong.
- "Why does `n_items` rank high in SHAP?" Orders with several items are dissatisfied more often even when on time (shown in the BI analysis); possible partial deliveries, but the data cannot confirm the reason.
- "Repeat purchase shows no effect - does experience not matter?" Only about 3% of customers ever reorder, so the test has little power; I did not claim retention conclusions.
- "Why did you not tune hyper-parameters?" To keep the comparison fair and the test set untouched; fixed sensible values, early stopping on validation for XGBoost.

## 6. Live demo script (about 8 minutes)
1. `python run.py`, open the Dashboard: explain KPIs, late vs negative-review trend, delay-band chart ("62% dissatisfied when late").
2. Data Management: database connected, funnel from 99,441 to 95,824 orders, upload `olist_sellers_dataset.csv` and then a deliberately bad file to show validation.
3. Review Intelligence: paste "Produto nao chegou ate hoje e ninguem responde"; show categories, severity, aspects, model sentiment; show the reliability box.
4. Prediction: predict with and without actual delivery days (e.g. 35); explain score vs observed rate.
5. Explainable AI: factor-group chart, beeswarm, explain the "highest risk" test order.
6. Sellers/Products/Geography: segments and hotspots, confidence intervals, minimum orders.
7. Research & Recommendations: verdict table, honest limitations, recommendation R1.
Backup: all figures are in `docs/figures`, the report in `docs/PROJECT_REPORT.docx`.

## ECX-Intelligence 2.0 viva additions
- **Why is checkout-time late prediction weak?** Lateness came in surges (Nov 2017, Feb-Mar 2018) caused by factors not in the data; test period late rate is 4.2% vs 7.7% in training (distribution shift). Honest AUC 0.65; it improves to 0.79 once handling times are known.
- **Why did you change the model-selection rule?** XGBoost was early-stopped on the validation set, so its validation score was optimistic. I changed the rule after seeing this and disclosed it; all models are still reported.
- **Is an anomaly fraud?** No. It is a statistical signal (FDR-controlled test), to be checked by a human.
- **Do your agents use an LLM?** No. Deterministic Supervisor + 5 agents calling pandas/scipy tools; fewer failure modes, fully testable. An optional LLM can only rephrase, and is rejected if it adds numbers.
- **How do you prove agents do not hallucinate?** Every finding carries an evidence ID; 30 sellers were compared with an independent computation (30/30 agree). That shows faithfulness, not causal truth.
- **What is not done?** Human studies, multilingual transformer, real-LLM tests (see docs/V2_EXPERIMENT_PROTOCOLS.md).
- **Simulation caveat:** it is hypothetical and associative, not a causal forecast.
