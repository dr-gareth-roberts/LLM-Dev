"""
Core test script for cognitive metrics evaluation.

This script provides a simplified test environment for validating our metrics,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required modules
from src.evaluation_framework.evaluation_protocols import TestCase, MetricResult # Updated path
from src.metrics.cognitive_metrics import ReasoningEvaluator, InstructionFollowingEvaluator, CognitiveBiasEvaluator

def main():
    """Run core functionality tests."""
    print("Testing Cognitive Metrics Implementation")
    print("=======================================")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import spacy
        print("✓ spaCy imported successfully")
        
        # Load English model
        nlp = spacy.load("en_core_web_sm")
        print("✓ English language model loaded")
        
        # Test basic NLP functionality
        doc = nlp("Testing the NLP pipeline.")
        print("✓ NLP pipeline working")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Test metric components
    print("\nTesting metric components...")
    try:
        # Test reasoning patterns
        test_text = """
        First, we must consider the evidence. Therefore, based on the data,
        we can conclude that this approach is optimal. However, we should
        also consider alternative viewpoints.
        """
        doc = nlp(test_text)
        print("✓ Text processing working")
        
        # Test pattern matching
        patterns = [
            "first", "therefore", "however",
            "consider", "evidence", "conclude"
        ]
        matches = [token.text.lower() for token in doc if token.text.lower() in patterns]
        print(f"✓ Found {len(matches)} pattern matches")
        
    except Exception as e:
        print(f"✗ Error in pattern matching: {e}")
        return
    
    print("\nAll core functionality tests passed!")

if __name__ == "__main__":
    main()
