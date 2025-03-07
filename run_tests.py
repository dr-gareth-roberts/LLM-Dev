"""
Test runner script for cognitive metrics evaluation.

This script sets up the test environment and runs the test suite,
following British English standards and proper type safety.
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the Python environment for testing."""
    project_root = Path(__file__).parent
    python_exe = r"C:\Users\PC\anaconda3\python.exe"
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root)
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install spaCy model
    print("Installing spaCy model...")
    subprocess.run([python_exe, "-m", "spacy", "download", "en_core_web_sm"])
    
    # Run tests
    print("Running tests...")
    subprocess.run([python_exe, "-m", "pytest", "tests/test_cognitive_metrics.py", "-v"])

if __name__ == "__main__":
    setup_environment()
