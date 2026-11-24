# One command that runs every quality gate on Windows (PowerShell):   .\scripts\check.ps1
# Install the tools once:   pip install ruff bandit pip-audit pytest-cov
$ErrorActionPreference = "Stop"
Write-Host "1/4 Lint";      ruff check .
Write-Host "2/4 Security";  bandit -q -r app ecx2 ecx3 multids recsys -ll
Write-Host "3/4 Packages";  pip-audit -r requirements.txt
Write-Host "4/4 Tests";     python -m pytest -q --cov --cov-report=term
Write-Host "All checks finished."
