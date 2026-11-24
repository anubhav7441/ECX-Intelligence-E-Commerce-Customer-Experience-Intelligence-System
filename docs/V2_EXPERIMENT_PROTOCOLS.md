# Protocols for experiments that could NOT be run in the build environment
Do not report numbers for these until you have measured them yourself.

## RQ5 - do agents save analyst time? (human study)
1. Recruit 5+ classmates; give each 4 sellers (2 with anomaly flags, 2 without).
2. Manual condition: answer "what is wrong with this seller and what would you do?" using only the v1 dashboard; record minutes.
3. Agent condition: same question using the AI Investigation Center on different sellers (counterbalance order).
4. Two independent raters score answers 1-5 for correctness and actionability. Report mean time difference with a paired test and confidence interval; report rater agreement.

## RQ7 - does grounding reduce LLM hallucination?
1. Install Ollama and a small model; set `ECX_LLM_URL=http://localhost:11434`.
2. Write 30 questions with answers computed by you from the data.
3. Condition A: raw LLM, no data. Condition B: assistant with grounding. Score numeric correctness and unsupported claims. Report counts.

## Multilingual NLP (Experiment 3)
See `ecx2/nlp/experiments.md`.

## Explainability user study (RQ6)
Show SHAP vs no-SHAP explanations to participants; measure trust and decision accuracy with a short questionnaire.
