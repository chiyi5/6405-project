@echo off
cd /d "%~dp0..\.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" showcase_app.py --open-browser
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" showcase_app.py --open-browser
) else (
  python showcase_app.py --open-browser
)
pause
