@echo off
REM Smoke test only (~2-4 min). Run this FIRST on the lab PC.
setlocal
cd /d "%~dp0\.."
set PYTHONUNBUFFERED=1
python scripts\run_all.py --quick --skip-thermal %*
if errorlevel 1 (
  echo SMOKE TEST FAILED.
  pause
  exit /b 1
)
echo Smoke test finished. If that looked healthy, run scripts\run_all.bat for the full job.
pause
