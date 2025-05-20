# PowerShell script to set up environment and run tests
# Following British English standards

# Stop on any error
$ErrorActionPreference = "Stop"

Write-Host "`nSetting up Test Environment" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Function to validate Python installation
function Test-PythonInstallation {
    try {
        Write-Host "`nChecking Python installation..." -ForegroundColor Green
        $pythonVersion = & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -c "import sys; print(sys.version)"
        Write-Host "✓ Python version: $pythonVersion"
        return $true
    }
    catch {
        Write-Host "✗ Python check failed: $_" -ForegroundColor Red
        return $false
    }
}

# Function to validate dependencies
function Test-Dependencies {
    try {
        Write-Host "`nChecking dependencies..." -ForegroundColor Green
        
        # Check spaCy
        Write-Host "Checking spaCy..." -NoNewline
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -c "import spacy; print(f'✓ Version {spacy.__version__}')"
        
        # Check pytest
        Write-Host "Checking pytest..." -NoNewline
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -c "import pytest; print(f'✓ Version {pytest.__version__}')"
        
        # Check pytest-asyncio
        Write-Host "Checking pytest-asyncio..." -NoNewline
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -c "import pytest_asyncio; print('✓ Installed')"
        
        return $true
    }
    catch {
        Write-Host "`n✗ Dependency check failed: $_" -ForegroundColor Red
        return $false
    }
}

# Function to run validation tests
function Test-CoreFunctionality {
    try {
        Write-Host "`nRunning core functionality tests..." -ForegroundColor Green
        
        # Set PYTHONPATH to project root
        $env:PYTHONPATH = (Join-Path $PSScriptRoot "..")
        
        # Run our minimal test script
        Write-Host "Running environment test..."
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" (Join-Path $PSScriptRoot "..\tests\test_env.py") # Updated path
        
        # Run pattern matching test
        Write-Host "`nRunning pattern matching test..."
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" (Join-Path $PSScriptRoot "..\tests\test_patterns.py") # Updated path
        
        return $true
    }
    catch {
        Write-Host "`n✗ Core functionality test failed: $_" -ForegroundColor Red
        return $false
    }
}

# Main execution
try {
    # Check Python
    if (-not (Test-PythonInstallation)) {
        throw "Python installation check failed"
    }
    
    # Check dependencies
    if (-not (Test-Dependencies)) {
        Write-Host "`nInstalling missing dependencies..." -ForegroundColor Yellow
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -m pip install -r (Join-Path $PSScriptRoot "..\config\requirements.txt") # Updated path
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -m spacy download en_core_web_sm
    }
    
    # Run core tests
    if (-not (Test-CoreFunctionality)) {
        throw "Core functionality tests failed"
    }
    
    Write-Host "`n✓ All checks completed successfully!" -ForegroundColor Green
    
}
catch {
    Write-Host "`n✗ Setup failed: $_" -ForegroundColor Red
    Write-Host "`nTraceback:" -ForegroundColor Red
    Write-Host $_.Exception.ToString()
    exit 1
}
