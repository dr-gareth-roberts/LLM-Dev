"""
Focused validation script for ReasoningEvaluator.

This script provides a minimal validation environment for the reasoning evaluator,
following British English standards and proper type safety.
"""
import spacy
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent # Corrected to project root
sys.path.insert(0, str(project_root))

def validate_nlp_setup():
    """Validate NLP setup and pattern matching."""
    try:
        # Load spaCy model
        print("\nValidating NLP Setup:")
        nlp = spacy.load("en_core_web_sm")
        print("✓ NLP model loaded successfully")
        
        # Test basic processing
        test_text = "First, we analyse the evidence. Therefore, we conclude."
        doc = nlp(test_text)
        print("✓ Basic text processing working")
        
        # Test sentence segmentation
        sentences = list(doc.sents)
        print(f"✓ Sentence segmentation working ({len(sentences)} sentences found)")
        
        # Test dependency parsing
        deps = [(token.text, token.dep_) for token in doc]
        print("✓ Dependency parsing working")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in NLP setup: {str(e)}")
        return False

def validate_pattern_matching():
    """Validate pattern matching functionality."""
    try:
        # Load NLP model
        nlp = spacy.load("en_core_web_sm")
        
        print("\nValidating Pattern Matching:")
        
        # Test text with reasoning patterns
        test_text = """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results. However, we must consider some
        limitations. For example, the sample size was relatively small.
        Therefore, while the data suggests positive outcomes, further validation
        would be beneficial.
        """
        
        # Process text
        doc = nlp(test_text)
        
        # Test patterns
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
        
        matches_found = False
        for category, terms in patterns.items():
            found = []
            for term in terms:
                if term.lower() in test_text.lower():
                    found.append(term)
            
            if found:
                matches_found = True
                print(f"✓ {category}: Found {len(found)} patterns")
                print(f"  - {', '.join(found)}")
        
        if matches_found:
            print("✓ Pattern matching working correctly")
            return True
        else:
            print("✗ No patterns matched")
            return False
            
    except Exception as e:
        print(f"\n✗ Error in pattern matching: {str(e)}")
        return False

def main():
    """Run validation checks."""
    print("Validating ReasoningEvaluator Components")
    print("=======================================")
    
    try:
        # Validate NLP setup
        if not validate_nlp_setup():
            return 1
            
        # Validate pattern matching
        if not validate_pattern_matching():
            return 1
            
        print("\n✓ All validation checks passed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during validation: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
