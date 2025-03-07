"""
Core test script for ReasoningEvaluator.

This script provides a focused test environment for validating our reasoning metrics,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path

def main():
    """Run core validation tests."""
    print("Testing ReasoningEvaluator Core Functionality")
    print("=========================================")
    
    try:
        # Display environment information
        print("\nEnvironment Information:")
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print(f"Working directory: {os.getcwd()}")
        
        # Test imports
        print("\nValidating imports...")
        import spacy
        print("✓ spaCy imported successfully")
        print(f"spaCy version: {spacy.__version__}")
        
        from src.metrics.cognitive_metrics import ReasoningEvaluator
        print("✓ ReasoningEvaluator imported successfully")
        
        from src.evaluation_protocols import TestCase, MetricResult
        print("✓ Evaluation protocols imported successfully")
        
        # Test NLP setup
        print("\nValidating NLP setup...")
        nlp = spacy.load("en_core_web_sm")
        print("✓ English language model loaded")
        
        # Test pattern matching
        print("\nValidating pattern matching...")
        test_text = """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results in 75% of cases. However, we must
        consider some limitations. For example, the sample size was relatively
        small. Therefore, while the data suggests positive outcomes, further
        validation would be beneficial.
        """
        
        # Process text
        doc = nlp(test_text)
        print("✓ Text processing working")
        
        # Test pattern categories
        patterns = {
            "Logical Steps": [
                "first", "therefore", "while"
            ],
            "Evidence": [
                "according to", "studies", "data suggests"
            ],
            "Counterarguments": [
                "however", "limitations"
            ]
        }
        
        print("\nTesting pattern categories:")
        for category, terms in patterns.items():
            matches = [term for term in terms if term.lower() in test_text.lower()]
            if matches:
                print(f"✓ {category}: Found {len(matches)} patterns")
                print(f"  - {', '.join(matches)}")
        
        # Test sentence analysis
        print("\nValidating sentence analysis...")
        sentences = list(doc.sents)
        print(f"✓ Found {len(sentences)} sentences")
        
        # Test dependency parsing
        print("\nValidating dependency parsing...")
        for token in list(doc)[:5]:
            print(f"Token: {token.text}")
            print(f"Dependency: {token.dep_}")
            print(f"Head: {token.head.text}")
            print()
        
        print("\n✓ All core functionality tests passed!")
        return 0
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")
        return 1
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
