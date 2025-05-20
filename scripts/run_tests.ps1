# PowerShell script to set up test environment and run tests
# Following British English standards and proper type safety

$ErrorActionPreference = "Stop"

# Set environment variables
$env:PYTHONPATH = Join-Path $PSScriptRoot ".." # Corrected to project root
$pythonCmd = "C:\Users\PC\anaconda3\python.exe"

Write-Host "Setting up test environment..." -ForegroundColor Cyan

# Create and activate virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
& $pythonCmd -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
& $pythonCmd -m pip install -r (Join-Path $PSScriptRoot "..\config\requirements.txt") # Updated path

# Install spaCy model
Write-Host "Installing spaCy model..." -ForegroundColor Green
& $pythonCmd -m spacy download en_core_web_sm

# Run tests
Write-Host "Running tests..." -ForegroundColor Green
& $pythonCmd -m pytest tests/test_cognitive_metrics.py -v
