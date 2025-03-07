"""
Validation script for cognitive metrics framework.

This script provides a focused validation environment for our metrics,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path

def validate_environment():
    """Validate Python environment and dependencies."""
    try:
        print("\nValidating Environment:")
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print(f"Current directory: {os.getcwd()}")
        return True
    except Exception as e:
        print(f"✗ Environment validation failed: {e}")
        return False

def validate_imports():
    """Validate core module imports."""
    try:
        print("\nValidating Imports:")
        
        print("Importing spaCy...", end=" ")
        import spacy
        print("✓")
        
        print("Importing core modules...", end=" ")
        from src.metrics.cognitive_metrics import ReasoningEvaluator
        from src.evaluation_protocols import TestCase, MetricResult
        print("✓")
        
        return True
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False

def validate_nlp():
    """Validate NLP functionality."""
    try:
        print("\nValidating NLP:")
        
        import spacy
        print("Loading English model...", end=" ")
        nlp = spacy.load("en_core_web_sm")
        print("✓")
        
        test_text = "First, we analyse. Therefore, we conclude."
        print("Testing text processing...", end=" ")
        doc = nlp(test_text)
        print("✓")
        
        return True
    except Exception as e:
        print(f"\n✗ NLP validation failed: {e}")
        return False

def main():
    """Run validation checks."""
    print("Cognitive Metrics Framework Validation")
    print("====================================")
    
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Run validation checks
        checks = [
            ("Environment", validate_environment),
            ("Imports", validate_imports),
            ("NLP", validate_nlp)
        ]
        
        failed = False
        for name, check in checks:
            print(f"\nRunning {name} validation...")
            if not check():
                print(f"✗ {name} validation failed")
                failed = True
                break
        
        if not failed:
            print("\n✓ All validation checks passed!")
            return 0
        return 1
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
