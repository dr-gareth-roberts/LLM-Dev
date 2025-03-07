@echo off
REM Setup script for LLM-Dev environment
REM Following British English standards

echo Setting up Python environment...

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
python -m pip install -r requirements.txt

REM Install spaCy model
python -m spacy download en_core_web_sm

REM Run validation script
python check_nlp.py

echo Setup complete!
