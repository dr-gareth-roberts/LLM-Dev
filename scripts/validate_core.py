"""
Core validation script for cognitive metrics.

This script validates the core functionality of our cognitive metrics framework,
following British English standards and proper type safety.
"""
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to Python path for direct script execution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def validate_environment() -> Tuple[bool, Optional[str]]:
    """Validate Python environment and core dependencies.
    
    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    try:
        import spacy
        from src.metrics.cognitive_metrics import ReasoningEvaluator
        from src.evaluation_framework.evaluation_protocols import TestCase, MetricResult # Updated path
        
        return True, None
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def validate_nlp() -> Tuple[bool, Optional[str]]:
    """Validate NLP functionality.
    
    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test basic processing
        doc = nlp("First, we analyse. Therefore, we conclude.")
        sentences = list(doc.sents)
        
        if not sentences:
            return False, "No sentences found in test text"
            
        return True, None
    except Exception as e:
        return False, f"NLP error: {e}"

def validate_patterns() -> Tuple[bool, Optional[str], Dict[str, List[str]]]:
    """Validate pattern matching functionality.
    
    Returns:
        Tuple[bool, Optional[str], Dict[str, List[str]]]: 
            Success status, error message if any, and matched patterns
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        test_text = """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results. However, we must consider some
        limitations. For example, the sample size was relatively small.
        Therefore, while the data suggests positive outcomes, further validation
        would be beneficial.
        """
        
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
        return False, f"Pattern matching error: {e}", {}

def main() -> int:
    """Run core validation checks.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("Core Functionality Validation")
    print("===========================")
    
    try:
        # Validate environment
        print("\nValidating environment...")
        env_ok, env_error = validate_environment()
        if not env_ok:
            print(f"✗ Environment validation failed: {env_error}")
            return 1
        print("✓ Environment validated")
        
        # Validate NLP
        print("\nValidating NLP functionality...")
        nlp_ok, nlp_error = validate_nlp()
        if not nlp_ok:
            print(f"✗ NLP validation failed: {nlp_error}")
            return 1
        print("✓ NLP functionality validated")
        
        # Validate patterns
        print("\nValidating pattern matching...")
        patterns_ok, patterns_error, matches = validate_patterns()
        if not patterns_ok:
            print(f"✗ Pattern matching failed: {patterns_error}")
            return 1
            
        print("✓ Pattern matching validated")
        print("\nMatched Patterns:")
        for category, found in matches.items():
            if found:
                print(f"\n{category}:")
                for pattern in found:
                    print(f"- {pattern}")
        
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
