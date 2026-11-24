# Technology Stack
| Area | Technology |
|---|---|
| Language / server | Python 3.12+, Flask (app factory, blueprints), Werkzeug |
| Data | pandas, NumPy, PyArrow/Parquet, openpyxl (Excel) |
| Databases | SQLite: SQLAlchemy (v1 Olist), sqlite3 (accounts, ledger) |
| ML / stats | scikit-learn, XGBoost, SHAP, SciPy, statsmodels |
| Validation / reports | Pydantic 2, ReportLab (PDF) |
| Security | Werkzeug scrypt password hashing, signed cookie sessions, CSRF tokens, CSP and security headers |
| Front-end | Jinja2 templates, Bootstrap 5, Chart.js, plain JavaScript (all vendored; no CDN needed) |
| Typography | Inter variable font (SIL Open Font License), bundled locally |
| Tests | pytest (Flask test client), Playwright (browser journey checks) |
Exact versions: `requirements.txt`.
