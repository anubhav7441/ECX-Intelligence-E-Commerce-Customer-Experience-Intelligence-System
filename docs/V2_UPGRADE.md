# ECX-Intelligence 2.0 - what was added, how it works, what was measured

Status labels used throughout: **MEASURED** (script output in `v2_results/`), **TESTED** (automated tests in `tests/test_v2.py`), **UNTESTED** (code exists, not verifiable in the build environment), **NOT RUN**.

## 1. Modules and where they live
| Module | Code | UI page | Status |
|---|---|---|---|
| Early late-delivery risk (Workflow A) | `ecx2/risk/` | Predictive Risk Center | MEASURED, TESTED |
| Anomaly detection | `ecx2/anomaly/detect.py` | Anomaly Center | MEASURED (no ground truth), TESTED (planted-anomaly recovery) |
| Review intelligence v2 | `ecx2/nlp/` | Customer Voice 2.0 | MEASURED on 150 labels (single AI annotator), TESTED |
| Evidence ledger + Action Center | `ecx2/store.py` | Action Center & Evidence | TESTED |
| Multi-agent investigation | `ecx2/agents/` | AI Investigation Center | MEASURED vs independent gold, TESTED |
| Grounded research assistant | `ecx2/assistant/core.py`, `ecx2/tools.py` | AI Research Assistant | TESTED; optional LLM adapter UNTESTED with a real LLM |
| What-if simulation | `ecx2/sim/engine.py` | Business Simulation | TESTED (hypothetical, associative) |
| Evaluation center | `app/routes/v2.py` | Model Evaluation Center | shows measured results only |
| SHAP | v1 (`ml/explainability`) | Explainable AI (SHAP) | unchanged |

## 2. Workflow A - early late-delivery risk (measured)
Target `is_late`. Features known before delivery only (no delivery/delay/transit days, no review data). Historical seller/category/state late rates use only orders **delivered before** each purchase (unit-tested). Platform signals: trailing-30-day late rate and 7-day order volume. Temporal 70/15/15 split (test: 2018-06-21 to 2018-08-29, 4.2% late; train 7.7%).

| Stage | Model (selected) | Test ROC-AUC | PR-AUC | Top-decile lift | Observed late rate Low / Medium / High |
|---|---|---|---|---|---|
| A0 checkout | Logistic Regression | 0.652 | 0.069 | 1.5x | 1.4% / 4.5% / 6.6% |
| A1 carrier hand-off | Logistic Regression | 0.790 | 0.183 | 3.6x | 0.7% / 1.7% / 9.1% |

**Honest reading:** predicting lateness at checkout is weak; it becomes useful once approval and seller handling times are known. RF and XGBoost did *worse* on the test period (A0 ROC-AUC 0.42-0.43, below chance) because late deliveries came in surges (Nov 2017, Feb-Mar 2018) that do not repeat in the test period (distribution shift). **Model-selection note:** XGBoost is early-stopped on the validation set, so its validation score is optimistic; I changed the selection rule to "best validation PR-AUC among models not tuned on validation (LR, RF)" *after* seeing this. All models are reported in full; the rule change is disclosed on the page and in `risk_results.json`. Scores are rankings, not probabilities; use the observed-rate bands.

## 3. Anomaly detection (measured, no ground truth)
212 signals (141 sellers + 3 platform months): freight burden 82, seller dissatisfaction 35, sudden monthly negative change 34, late rate 24, Isolation Forest 19, seller-complaint frequency 15, platform months 3 (Nov 2017 late and negative-review spikes). Tests are one-sided binomial / Fisher with Benjamini-Hochberg FDR at 5%, minimum 30 orders per seller. There are no anomaly labels, so detection accuracy is **not** reported; a unit test checks that a planted anomaly among 40 normal sellers is recovered with few false positives. Wording is non-accusatory ("investigation signal"). Scores are comparable within a detector only.

## 4. Review intelligence v2
Adds main issue (priority order), evidence keywords (matched trigger terms), root-cause *hypothesis*, language check (English flagged unreliable), heuristic confidence (not a probability), and corpus-level distinctive TF-IDF terms per category. Main issue falls inside the labelled categories for 63% of 117 labelled reviews with an issue. **Multilingual transformer experiment: NOT RUN** (model downloads blocked); protocol in `ecx2/nlp/experiments.md`.

## 5. Multi-agent system
Supervisor -> Seller Investigation -> Delivery Risk -> Customer Review -> Research Validation -> Action Planning. Each returns a Pydantic `AgentReport` (task, data sources, tool calls, findings with evidence IDs, uncertainty, errors, next step). Agents never compute numbers themselves; they call `ecx2/tools.py`. A failing tool yields a `partial` report and the others still run (tested). Hypotheses: H1 excess dissatisfaction vs the seller's category mix; H2 late deliveries drive the seller's dissatisfaction (also reports the share of dissatisfaction that late orders explain). Evaluation on 30 random sellers vs an independent pandas implementation: 30/30 late-rate, 30/30 dissatisfied-rate, 30/30 verdicts agree, 235/235 findings have ledger IDs, 29/29 proposed actions link to valid evidence, 0 failures, 0.44 s per investigation. Because the gold code uses the same definitions, this shows **faithful reporting**, not that conclusions are causally right. **Manual-vs-agent time and human usefulness: NOT RUN.**

## 6. Human-in-the-loop
Actions start `proposed`. Only a named reviewer can approve/reject; invalid transitions, empty names and missing completion notes are rejected; every change is stored in `action_history`. The software never performs real-world actions.

## 7. Assistant
Rule-based intent router -> tools -> deterministic answer with citations; unsupported questions are refused with a list of supported types. Logged in `assistant_log`. Optional Ollama rephrasing is rejected if it introduces any number not present in the computed answer; LLM errors fall back silently (tested with stub functions only). **RQ7 (grounded vs ungrounded LLM): NOT RUN.**

## 8. What-if simulator
Re-scores test-period orders with the v1 delivery-time satisfaction model after changing delivery days, fixing a share of late orders, adding an estimate buffer or discounting freight. Output is labelled HYPOTHETICAL, uses isotonic calibration on test scores and bootstrap intervals over orders. Example: fixing all late orders lowers the expected dissatisfied rate by about 1.6 percentage points in the test cohort. Associative model, not causal.

## 9. Known limitations
One marketplace, 2016-2018, reviewed delivered orders only; weak checkout-time late prediction; temporal shift; single-annotator NLP labels; no human studies; no local LLM tested; anomaly accuracy unknown.

## 10. Run it
```powershell
pip install -r requirements.txt
python run.py                      # new pages are in the sidebar under "2.0 modules"
python -m pytest tests -q          # 101 tests (needs the full rebuild of v1 data)
python scripts\run_v2.py           # optional: rebuild all 2.0 artifacts (after scripts\run_all.py)
```
