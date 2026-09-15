@echo off
REM Double-click this, or run from Command Prompt / PowerShell inside hw2_5
setlocal
cd /d "%~dp0\.."
set PYTHONUNBUFFERED=1

where python >nul 2>&1
if errorlevel 1 (
  echo python is not on PATH. Open "Anaconda Prompt" or the lab CUDA environment, then:
  echo   python scripts\run_all.py
  pause
  exit /b 1
)

python scripts\run_all.py %*
if errorlevel 1 (
  echo.
  echo RUN FAILED. Do not start the 20-minute job until check_env / the smoke test works.
  pause
  exit /b 1
)

echo.
echo Finished. Copy results\ off this machine. Fill reservations\GPU_HOURS.md.
pause
