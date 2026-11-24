# Multi-Dataset Intelligence & Automated Report Generation (module M1-M8)

Open **Multi-Dataset Analysis** in the sidebar. Original Olist workflow, models and results are not modified (tested).

## What it does
1. Choose Olist (read-only, Dataset A) or upload a CSV / XLSX / JSONL(.gz) file (Online Retail II, Amazon Reviews 2023 sample, or any other).
2. Inspect: dimensions, column kinds (numeric/categorical/date/text/id), missing values, duplicates, quality score and issues.
3. Confirm or correct the automatic column mapping (e.g. "InvoiceDate -> date").
4. Pick analyses. The capability engine only enables analyses whose required column roles are mapped, and explains why the others are blocked.
5. Generate a 14-section evidence-based PDF report plus supporting CSV exports.

## Analyses
Sales & revenue, product, customer segmentation (RFM), geography, reviews (sentiment, complaints, aspects; English rules, Portuguese via v1 rules), delivery, seller comparison, numeric statistics + Isolation-Forest anomaly signals.

## Isolation and integrity
- Each dataset has its own id and folder `data/datasets/<id>/` (meta, processing history, cleaned data, results, report).
- Olist is a read-only registration of the existing v1 file; nothing is written to v1 files or models.
- The Olist-trained model is never applied to another dataset; the page explains why. No retraining is offered in this version.
- Report wording separates OBSERVED facts, ASSOCIATIONS (statistical, not causal) and SUGGESTIONS (hypotheses). Every number is computed.

## Adapters (cleaning, every step counted in the history)
- **Online Retail II:** duplicates removed; cancellations ("C" invoices) flagged and kept as negative sales; price <= 0 removed; non-product stock codes (POST, M, fees) removed; negative quantities counted; missing customer IDs kept for sales but excluded from customer analysis; Revenue = Quantity x Price. Two-sheet Excel files are stacked.
- **Amazon Reviews 2023:** timestamp (ms) converted; ratings validated 1-5; duplicates removed; HTML stripped; nested columns dropped; category and sample size recorded in the dataset metadata and report.

## Tested vs untested
- TESTED (18 automated tests + browser test): upload, profiling, mapping, capability gating, adapters, all analyses, report/PDF, isolation, security (path traversal, id validation), errors.
- Tested ONLY on SYNTHETIC files that copy the formats (`samples/`, `scripts/make_synthetic_samples.py`). **The real Online Retail II and Amazon files could not be downloaded in the build environment, so they are untested.** Download them yourself and try; if a column name differs, use the mapping step.
- Not included: model retraining on new datasets, legacy .xls files, Amazon product-metadata files.

## Limits
Files up to 80 MB and 2,000,000 rows; text analysis samples 30,000 reviews; Amazon sample up to 100,000 JSON lines.
