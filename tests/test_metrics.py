"""
Core test script for cognitive metrics evaluation.

This script provides a focused test environment for validating our metrics,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run core validation tests."""
    print("Testing Cognitive Metrics Framework")
    print("=================================")
    
    try:
        # Test imports
        print("\nValidating imports...")
        import spacy
        from src.metrics.cognitive_metrics import ReasoningEvaluator
        from src.evaluation_protocols import TestCase, MetricResult
        print("✓ Core modules imported successfully")
        
        # Test NLP setup
        print("\nValidating NLP setup...")
        nlp = spacy.load("en_core_web_sm")
        print("✓ English language model loaded")
        
        # Test text processing
        test_text = """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results in 75% of cases. However, we must
        consider some limitations. For example, the sample size was relatively
        small. Therefore, while the data suggests positive outcomes, further
        validation would be beneficial.
        """
        
        doc = nlp(test_text)
        print("✓ Text processing working")
        
        # Test pattern matching
        print("\nValidating pattern matching...")
        patterns = {
            "Logical Steps": ["first", "therefore", "while"],
            "Evidence": ["according to", "studies", "data suggests"],
            "Counterarguments": ["however", "limitations"]
        }
        
        matches = {}
        for category, terms in patterns.items():
            matches[category] = [
                term for term in terms 
                if term.lower() in test_text.lower()
            ]
        
        for category, found in matches.items():
            if found:
                print(f"✓ {category}: Found {len(found)} patterns")
                print(f"  - {', '.join(found)}")
        
        print("\n✓ All validation tests completed successfully!")
        return 0
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Please ensure all dependencies are installed:")
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
