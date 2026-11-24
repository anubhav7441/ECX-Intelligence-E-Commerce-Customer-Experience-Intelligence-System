# Presentation Outline (about 15 minutes, 14 slides)

Build the slides in PowerPoint/Google Slides from this outline. Use the real figures in `docs/figures/` and screenshots in `docs/screenshots/`. Numbers are from `docs/PROJECT_REPORT.md`.

1. **Title** - ECX-Intelligence: E-Commerce Customer Experience Intelligence. Name, guide, institute.
2. **Problem** - A star rating does not explain *why* customers are unhappy. (Use `03_guest_demo` or landing screenshot.)
3. **Objectives and research questions** - RQ1-RQ5.
4. **Data** - Olist, 99,441 orders; 95,824 analysed; data issues found (non-unique review ids, 58.7% without text).
5. **System design** - `docs/diagrams/architecture.png`.
6. **Pipeline** - `docs/diagrams/pipeline.png`.
7. **Key finding 1** - Late 62.4% vs on-time 9.3% dissatisfied (`eda_dissatisfaction_by_delay.png`). Say: association, not proof of cause.
8. **Key finding 2** - Complaints are about delay and incomplete orders, not price (`bi_complaint_categories.png`).
9. **Sellers and regions** - `bi_seller_segments.png`, `bi_state_hotspots.png`.
10. **Prediction** - leakage-safe, time-based split; ROC-AUC 0.61 vs 0.71; algorithm choice matters little.
11. **Explainability** - SHAP (`shap_importance_S1_R.png`).
12. **The web app** - screenshots 04-09: Overview, Advice, My Data, Risk, Simulator, Assistant.
13. **Advice in plain English** - problem, cause, what to do (`05_insights`); works for uploaded datasets too.
14. **Limitations and future work** - single-annotator AI labels, moderate predictive power, local prototype, not production-ready. Thank you / questions.

Rehearsal tip: also read `docs/VIVA_PREPARATION.md`.
