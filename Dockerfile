# ECX-Intelligence container image (code only). Data, models and databases are mounted as volumes (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# libgomp is needed by XGBoost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-prod.txt ./
RUN pip install -r requirements.txt && pip install -r requirements-prod.txt

COPY . .

# run as a normal user, not root
RUN useradd --create-home --uid 10001 ecx && chown -R ecx:ecx /app
USER ecx

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# SECRET_KEY must be supplied at run time (see docker-compose.yml). 2 workers is enough for a prototype.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "wsgi:app"]
