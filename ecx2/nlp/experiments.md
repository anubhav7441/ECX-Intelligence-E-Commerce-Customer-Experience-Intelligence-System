# Experiment 3 (classical vs multilingual NLP) - protocol (NOT RUN in the build environment)
1. Same 150-review labelled sample + a larger sample labelled by the student.
2. Models: TF-IDF+LR (done, v1), rule taxonomy (done, v1), a multilingual sentence-embedding model (e.g. paraphrase-multilingual-MiniLM) + LR.
3. Metrics: macro-F1 per category, sentiment accuracy, inference time. Report with confidence intervals (bootstrap).
4. Requires internet access to download the model; then run it on your own machine and record real numbers. Do not report numbers you did not measure.
