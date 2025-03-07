"""
System check script for cognitive metrics framework.

This script provides basic validation of our system configuration,
following British English standards.
"""
import sys
print(f"Python Version: {sys.version}")
print(f"Python Path: {sys.executable}")

try:
    print("\nChecking core dependencies...")
    import spacy
    print(f"✓ spaCy version: {spacy.__version__}")
    
    print("\nChecking NLP model...")
    nlp = spacy.load("en_core_web_sm")
    print("✓ English model loaded")
    
    print("\nTesting basic processing...")
    doc = nlp("First, we analyse. Therefore, we conclude.")
    print(f"✓ Found {len(list(doc.sents))} sentences")
    
    print("\nAll checks passed successfully!")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)
