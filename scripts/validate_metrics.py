"""
Validation script for cognitive metrics.

This script validates our cognitive metrics implementation,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent # Corrected to project root
sys.path.insert(0, str(project_root))

def validate_imports():
    """Validate core module imports."""
    try:
        print("\nValidating Core Imports:")
        
        # Test spaCy
        print("Importing spaCy...", end=" ")
        import spacy
        print(f"✓ (version {spacy.__version__})")
        
        # Test core modules
        print("Importing cognitive metrics...", end=" ")
        from src.metrics.cognitive_metrics import (
            ReasoningEvaluator,
            InstructionFollowingEvaluator,
            CognitiveBiasEvaluator
        )
        print("✓")
        
        print("Importing evaluation protocols...", end=" ")
        from src.evaluation_framework.evaluation_protocols import ( # Updated path
            TestCase,
            MetricResult,
            MetricCategory
        )
        print("✓")
        
        return True
    except Exception as e:
        print(f"\n✗ Import validation failed: {e}")
        return False

def validate_nlp():
    """Validate NLP functionality."""
    try:
        print("\nValidating NLP Components:")
        
        # Load model
        print("Loading English model...", end=" ")
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓")
        
        # Test text processing
        test_text = """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results. However, we must consider some
        limitations. For example, the sample size was relatively small.
        Therefore, while the data suggests positive outcomes, further validation
        would be beneficial.
        """
        
        print("Processing test text...", end=" ")
        doc = nlp(test_text)
        print("✓")
        
        # Test pattern matching
        print("\nTesting Pattern Recognition:")
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
        
        for category, terms in patterns.items():
            matches = [term for term in terms if term.lower() in test_text.lower()]
            if matches:
                print(f"✓ {category}: Found {len(matches)} patterns")
                print(f"  - {', '.join(matches)}")
        
        return True
    except Exception as e:
        print(f"\n✗ NLP validation failed: {e}")
        return False

def validate_metrics():
    """Validate metrics implementation."""
    try:
        print("\nValidating Metrics Implementation:")
        
        # Import required modules
        from src.metrics.cognitive_metrics import ReasoningEvaluator
        from src.evaluation_framework.evaluation_protocols import TestCase # Updated path
        
        # Create mock environment
        class MockEnv:
            async def get_model_response(self, model_id, test_case):
                return (
                    "First, let's examine the evidence. According to studies, "
                    "the results are promising. However, we must consider "
                    "limitations. Therefore, further validation is needed."
                )
        
        # Create evaluator instance
        print("Initialising ReasoningEvaluator...", end=" ")
        evaluator = ReasoningEvaluator(MockEnv())
        print("✓")
        
        # Validate component structure
        print("Validating evaluator structure...", end=" ")
        assert hasattr(evaluator, '_extract_reasoning_components')
        assert hasattr(evaluator, '_analyse_reasoning_quality')
        print("✓")
        
        return True
    except Exception as e:
        print(f"\n✗ Metrics validation failed: {e}")
        return False

def main():
    """Run validation checks."""
    print("Cognitive Metrics Framework Validation")
    print("====================================")
    
    try:
        # Run validation checks
        checks = [
            ("Core Imports", validate_imports),
            ("NLP Components", validate_nlp),
            ("Metrics Implementation", validate_metrics)
        ]
        
        failed = False
        for name, check in checks:
            if not check():
                print(f"\n✗ {name} validation failed")
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
