<#
Create (or remove) a daily Windows Task Scheduler job that runs one ECX monitor cycle.
It runs:  python -m ecx3.scheduler --once   inside the project folder.

Usage (PowerShell, from the project folder):
  .\scripts\schedule_windows.ps1                      # daily at 09:00
  .\scripts\schedule_windows.ps1 -Time 18:30          # daily at 18:30
  .\scripts\schedule_windows.ps1 -Remove              # delete the task
Check it:   schtasks /Query /TN "ECX Monitor"
Run now:    schtasks /Run /TN "ECX Monitor"
If scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#>
param([string]$Time = "09:00", [string]$Name = "ECX Monitor", [switch]$Remove)

if ($Remove) { schtasks /Delete /TN $Name /F; exit $LASTEXITCODE }

if ($Time -notmatch '^([01]\d|2[0-3]):[0-5]\d$') { Write-Error "Time must look like 09:00 (24-hour)."; exit 1 }
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venv) { $venv } else { (Get-Command python -ErrorAction Stop).Source }
$cmd = "cmd /c cd /d `"$root`" && `"$py`" -m ecx3.scheduler --once >> logs\scheduler.log 2>&1"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
schtasks /Create /SC DAILY /ST $Time /TN $Name /TR $cmd /F
if ($LASTEXITCODE -eq 0) { Write-Host "Created '$Name' (daily at $Time). Test it with: schtasks /Run /TN `"$Name`"" }
exit $LASTEXITCODE
