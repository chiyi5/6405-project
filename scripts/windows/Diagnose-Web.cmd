@echo off
setlocal
cd /d "%~dp0..\.."
set "LEARNING_PYTHON=python"
if exist "%~dp0..\..\..\.venv\Scripts\python.exe" set "LEARNING_PYTHON=%~dp0..\..\..\.venv\Scripts\python.exe"
if exist "%~dp0..\..\.venv\Scripts\python.exe" set "LEARNING_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
echo Testing the actual web app and an independent minimal HTTP server...
"%LEARNING_PYTHON%" diagnose_web.py
pause
