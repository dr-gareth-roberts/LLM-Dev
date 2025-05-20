@echo off
set PYTHONPATH=%~dp0..
set PYTHON_CMD=C:\Users\PC\anaconda3\python.exe

echo Installing dependencies...
%PYTHON_CMD% -m pip install -r "%~dp0..\config\requirements.txt"

echo Installing spaCy model...
%PYTHON_CMD% -m spacy download en_core_web_sm

echo Running tests...
%PYTHON_CMD% -m pytest tests/test_cognitive_metrics.py -v
