"""
Core test script for cognitive metrics.

This script provides a focused test environment for our cognitive metrics,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

def validate_imports() -> Tuple[bool, Optional[str]]:
    """Validate core module imports.
    
    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    try:
        print("\nValidating imports...")
        
        # Test spaCy
        import spacy
        print(f"✓ spaCy version: {spacy.__version__}")
        
        # Test core modules
        from src.metrics.cognitive_metrics import (
            ReasoningEvaluator,
            InstructionFollowingEvaluator,
            CognitiveBiasEvaluator
        )
        print("✓ Cognitive metrics imported")
        
        from src.evaluation_protocols import (
            TestCase,
            MetricResult,
            MetricCategory
        )
        print("✓ Evaluation protocols imported")
        
        return True, None
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def validate_nlp() -> Tuple[bool, Optional[str], Dict[str, List[str]]]:
    """Validate NLP functionality and pattern matching.
    
    Returns:
        Tuple[bool, Optional[str], Dict[str, List[str]]]:
            Success status, error message if any, and matched patterns
    """
    try:
        print("\nValidating NLP functionality...")
        
        # Load model
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ English model loaded")
        
        # Test text with reasoning patterns
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
        
        matches: Dict[str, List[str]] = {}
        for category, terms in patterns.items():
            matches[category] = [
                term for term in terms 
                if term.lower() in test_text.lower()
            ]
        
        if not any(matches.values()):
            return False, "No patterns matched", {}
            
        return True, None, matches
    except Exception as e:
        return False, f"NLP error: {e}", {}

def validate_metrics() -> Tuple[bool, Optional[str]]:
    """Validate metrics implementation.
    
    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    try:
        print("\nValidating metrics implementation...")
        
        # Import required modules
        from src.metrics.cognitive_metrics import ReasoningEvaluator
        from src.evaluation_protocols import TestCase, LLMEnvironmentProtocol
        
        # Create mock environment
        class MockEnv(LLMEnvironmentProtocol):
            async def get_model_response(self, model_id: str, test_case: TestCase) -> str:
                return (
                    "First, let's examine the evidence. According to studies, "
                    "the results are promising. However, we must consider "
                    "limitations. Therefore, further validation is needed."
                )
        
        # Create evaluator instance
        evaluator = ReasoningEvaluator(MockEnv())
        print("✓ ReasoningEvaluator initialised")
        
        # Validate component structure
        assert hasattr(evaluator, '_extract_reasoning_components')
        assert hasattr(evaluator, '_analyse_reasoning_quality')
        print("✓ Component structure validated")
        
        return True, None
    except Exception as e:
        return False, f"Metrics validation failed: {e}"

def main() -> int:
    """Run validation tests.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("Cognitive Metrics Framework Validation")
    print("====================================")
    
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Run validation checks
        import_ok, import_error = validate_imports()
        if not import_ok:
            print(f"\n✗ Import validation failed: {import_error}")
            return 1
        
        nlp_ok, nlp_error, matches = validate_nlp()
        if not nlp_ok:
            print(f"\n✗ NLP validation failed: {nlp_error}")
            return 1
            
        print("\nPattern Matches:")
        for category, found in matches.items():
            if found:
                print(f"\n{category}:")
                for pattern in found:
                    print(f"- {pattern}")
        
        metrics_ok, metrics_error = validate_metrics()
        if not metrics_ok:
            print(f"\n✗ Metrics validation failed: {metrics_error}")
            return 1
        
        print("\n✓ All validation checks passed!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
