# ECX-Intelligence 2.0 - Baseline audit (Phase 0/1)

Audit of the working v1 project before the 2.0 upgrade. v1 files and models were **not modified or deleted** (only additive changes: new blueprint registered in `app/__init__.py`, new nav block in `base.html`, `APP_DB` setting in `config.py`, `pydantic` in requirements).

## What v1 contains (verified by running it)
| Area | State |
|---|---|
| Data | Olist, 99,441 orders; 95,824 delivered+reviewed analysed; order-level aggregation before joins |
| Statistics | 7 hypotheses with effect sizes (late vs on-time dissatisfied 62.4% vs 9.3%, OR about 16) |
| NLP | Rule-based Portuguese taxonomy (10 classes), aspect sentiment, severity, TF-IDF+LR sentiment |
| ML | Satisfaction (1-2 stars) models, 4 feature sets, temporal split, no leakage; best test ROC-AUC 0.61 checkout / 0.71 delivery-time |
| XAI | SHAP (grouped) |
| App | Flask, 10 pages, REST API, SQLite |
| Tests | 70 passing |

## Gaps the 2.0 prompt asks to close
Early *late-delivery* prediction, anomaly detection, main-issue/root-cause review output, multi-agent investigation, grounded assistant, evidence ledger, what-if simulation, action workflow with human approval, model evaluation page.

## Defects / risks found in the audit
1. **NLP word-boundary bug (v1)**: pattern `ma qualidade` also fired inside `otima qualidade`, labelling praise as a product-quality complaint. Kept in v1 for reproducibility of the v1 report; fixed in v2 (`ecx2/nlp/issues.py`). Aggregate effect on the 150-review sample is negligible (micro-F1 0.667 -> 0.664).
2. v1 complaint labels come from a single AI annotator, not independent human labels (unchanged, disclosed).
3. Only reviewed orders are analysed (selection bias) - applies to all 2.0 models too.

## Design decisions for 2.0
- New code lives in the `ecx2/` package + `app/routes/v2.py`; results in `v2_results/`; models in `models/v2/`; ledger/actions in a **separate** SQLite file `database/ecx_app.db`.
- Custom deterministic Supervisor + 5 agents with Pydantic outputs (no LangGraph/CrewAI: fewer dependencies, offline, fully testable).
- No LLM required anywhere. Optional local LLM (Ollama, `ECX_LLM_URL`) can only rephrase an answer already computed from data, and is rejected if it adds numbers.
