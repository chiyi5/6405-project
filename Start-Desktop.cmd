@echo off
setlocal
cd /d "%~dp0"
set "LEARNING_PYTHON=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "LEARNING_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "LEARNING_PYTHON=%~dp0.venv\Scripts\python.exe"
echo Starting SkillBridge desktop. No browser or network port is needed.
"%LEARNING_PYTHON%" learning_desktop.py
if errorlevel 1 pause
