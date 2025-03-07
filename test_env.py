"""
Environment test script for cognitive metrics.

This script validates the Python environment and core dependencies,
following British English standards.
"""
import sys
print(f"Python Version: {sys.version}")
print(f"Python Path: {sys.executable}")
print("\nTesting imports...")
try:
    import spacy
    print("✓ spaCy imported successfully")
except ImportError as e:
    print(f"✗ spaCy import failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
