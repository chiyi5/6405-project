@echo off
setlocal
cd /d "%~dp0"
set "LEARNING_PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "LEARNING_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "LEARNING_PYTHON=%~dp0.venv\Scripts\python.exe"
echo Starting SkillBridge at http://127.0.0.1:8765
echo Keep this window open while using the app. Press Ctrl+C to stop.
"%LEARNING_PYTHON%" learning_app.py --port 8765 --open-browser
echo.
echo The server has stopped. Any error message is shown above.
pause
