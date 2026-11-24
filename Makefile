# Shortcuts for Linux/macOS (Windows users: use scripts/check.ps1).   make check
.PHONY: run test lint security cov check docker
run:       ; python run.py
test:      ; python -m pytest -q
lint:      ; ruff check .
security:  ; bandit -q -r app ecx2 ecx3 multids recsys -ll && pip-audit -r requirements.txt
cov:       ; python -m pytest -q --cov --cov-report=term-missing
check: lint security cov
docker:    ; docker build -t ecx-intelligence .
