@echo off
rem Runs the experiments added in the 2026-11 upgrade, in order. Needs internet the first time
rem (HuggingFace models, GloVe). Results: results\learning\bench\ and results\step9\.
setlocal
cd /d "%~dp0..\.."
set "PY=python"
if exist "%~dp0..\..\..\.venv\Scripts\python.exe" set "PY=%~dp0..\..\..\.venv\Scripts\python.exe"
if exist "%~dp0..\..\.venv\Scripts\python.exe" set "PY=%~dp0..\..\.venv\Scripts\python.exe"
"%PY%" -m pip install -q rank-bm25 scikit-learn matplotlib
"%PY%" -m pip install -q gensim || echo [info] gensim not available for this Python version - Word2Vec will be skipped.
echo === 1/3 Learning module: CPU benchmark (L1-L3) ===
"%PY%" experiments\learning\run_all.py
echo === 2/3 Learning module: Sentence-BERT, cross-encoder, fine-tuning (L4) ===
"%PY%" experiments\learning\run_all.py --neural
echo === 3/3 Main project: traditional NLP vs transformers (step9) ===
"%PY%" experiments\step9_classic_vs_neural.py
echo.
echo Done. Send results\learning\bench and results\step9 back if anything failed.
pause
