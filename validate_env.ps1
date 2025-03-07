# PowerShell script to validate environment and run tests
# Following British English standards

Write-Host "`nValidating Python Environment"
Write-Host "=========================`n"

# Set environment variables
$env:PYTHONPATH = "c:\Users\PC\PycharmProjects\LLM-Dev"

# Check Python installation
Write-Host "Checking Python..."
try {
    $pythonVersion = & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -c "import sys; print(sys.version)"
    Write-Host "✓ Python found: $pythonVersion"
} catch {
    Write-Host "✗ Error checking Python: $_"
    exit 1
}

# Check core dependencies
Write-Host "`nChecking dependencies..."
$dependencies = @("spacy", "pytest", "pytest-asyncio")
foreach ($dep in $dependencies) {
    try {
        & "C:\Users\PC\anaconda3\envs\Everything\python.exe" -c "import $dep; print(f'✓ {$dep} version: ' + $dep.__version__)"
    } catch {
        Write-Host "✗ Error importing $dep"
        exit 1
    }
}

# Run validation script
Write-Host "`nRunning validation tests..."
try {
    & "C:\Users\PC\anaconda3\envs\Everything\python.exe" test_cognitive_core.py
} catch {
    Write-Host "✗ Error running tests: $_"
    exit 1
}
