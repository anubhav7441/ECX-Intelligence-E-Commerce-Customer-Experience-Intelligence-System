# Explaining Customer Dissatisfaction in E-Commerce: Delivery Delay, Seller History and Leakage-Safe Prediction on the Olist Dataset

*Draft for author review. Every number below is copied from `docs/PROJECT_REPORT.md`, which is generated from the experiment output files. Related-work citations beyond the methodological sources listed must be added by the author after reading the literature; none were invented here.*

## Abstract
We study what drives 1-2 star reviews in the public Olist Brazilian marketplace data (99,441 orders, 2016-2018; 95,824 delivered and reviewed orders analysed). 62.4% of late orders received a 1-2 star review versus 9.3% of on-time orders (odds ratio 16.3; Cliff's delta -0.64); the association shrinks (delta -0.30) when only reviews written after the recorded delivery are used. Under a time-based split with leakage controls, the best checkout-time model reaches ROC-AUC 0.61 and the best delivery-time model 0.71. Seller/category/state review history adds a small but reliable gain (+0.039 and +0.027 ROC-AUC, paired-bootstrap CIs exclude zero); algorithm choice matters little. SHAP ranks delivery delay first in the delivery-time model. Findings are associations, not causal effects.

## 1. Introduction
A review score does not say why an experience failed. We ask: RQ1 how delivery performance relates to satisfaction; RQ2 which aspects dominate negative reviews; RQ3 whether review-derived history helps prediction under valid settings; RQ4 which algorithm is best; RQ5 which factors SHAP identifies.

## 2. Related work (methodological sources used)
SHAP (Lundberg & Lee, 2017); XGBoost (Chen & Guestrin, 2016); random forests (Breiman, 2001); Mann-Whitney (1947) and Cliff's delta (1993); bootstrap (Efron, 1979); aspect-based sentiment task definition (Pontiki et al., 2016); Olist public dataset (Kaggle). *Domain literature on delivery performance and satisfaction: to be added by the author.*

## 3. Data
Nine CSV files verified. Review ids are not unique (latest review kept); items, payments and reviews are aggregated to order level before joining (no join inflation, checked against SQL); 58.7% of reviews have no text; 4,976 orders have a review dated before the recorded delivery (sensitivity analysis reported).

## 4. Method
Target: dissatisfied = 1-2 stars. Two prediction moments: checkout-time (S0) and delivery-time (S1). Review-derived features (R) use only reviews created strictly before the order's purchase and are shrunk to the training mean; a brute-force test confirms no own or future review. Time-ordered 70/15/15 split (test 2018-06-21 to 2018-08-29, n=14,374, 9.6% dissatisfied). Logistic Regression, Decision Tree, Random Forest, XGBoost with fixed hyper-parameters; threshold chosen on validation F1; test set evaluated once; paired bootstrap for differences. Statistics: Mann-Whitney with Cliff's delta, chi-square with odds ratio, Spearman, homogeneity tests with Cramer's V and Wilson intervals, logistic regression. NLP: transparent rule taxonomy (10 classes), lexicon aspect sentiment with negation, TF-IDF + LR sentiment; the complaint classifier was checked on 150 reviews labelled blind by a single AI annotator (not independent human gold labels).

## 5. Results
**RQ1.** Mean score 2.27 (late) vs 4.29 (on time); dissatisfaction rises from 12% (on the estimated day) to 80% (8-14 days late).
**RQ2.** Late/not-delivered appears in 40.3% of negative commented reviews; price and freight effects are negligible (Cliff's delta about -0.04).
**Sellers, categories, regions.** 52 of 413 sellers, 9 of 52 categories and 9 of 24 states are significantly worse than overall (small effect sizes for categories and states: Cramer's V 0.07 and 0.09).
**RQ3/RQ4.** Best test ROC-AUC: S0 0.570-0.580, S0+R 0.608, S1 0.687-0.693, S1+R 0.714 (XGBoost selected on validation PR-AUC). Deltas for adding history: +0.039 [0.026, 0.050] (checkout) and +0.027 [0.015, 0.039] (delivery). Delivery outcome (S1 vs S0): +0.117 [0.094, 0.138]. Algorithm differences are at most 0.014 ROC-AUC.
**Sensitivity.** Restricting to reviews written after delivery lowers delivery-time ROC-AUC from 0.714 to 0.630.
**RQ5.** Delivery-time model: delivery performance 57.6% of mean |SHAP|, order structure 23.1%, historical reviews 15.2%; top feature delay_days.
**Repeat purchase.** 3.77% after a dissatisfied first order vs 4.10% after a satisfied one (p=0.19, not significant).

## 6. Discussion and limitations
Delivery delay is the dominant actionable factor; structured data alone predicts dissatisfaction only moderately. Limits: observational data, review timing, selection bias (only reviewers), single-annotator AI labels, scores not calibrated probabilities, one marketplace, 2016-2018.

## 7. Conclusion
Reduce late deliveries (especially carrier transit), monitor weak sellers, and treat multi-item orders as higher risk. Future work: independent human labels, other marketplaces, causal designs.

## Reproducibility
Code, tests, figures and result files are in the project; see `README.md` and `docs/TESTING_REPORT.md`.
