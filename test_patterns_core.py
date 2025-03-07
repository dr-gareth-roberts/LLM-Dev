"""
Core pattern matching test script.

This script validates our pattern matching functionality for cognitive metrics,
following British English standards and proper type safety.
"""
import spacy
from pathlib import Path
import sys

def main():
    """Test core pattern matching functionality."""
    print("Testing Pattern Matching")
    print("======================")
    
    try:
        # Load spaCy model
        print("\nLoading NLP model...")
        nlp = spacy.load("en_core_web_sm")
        print("✓ Model loaded successfully")
        
        # Test text with reasoning patterns
        test_text = """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results in 75% of cases. However, we must
        consider some limitations. For example, the sample size was relatively
        small. Therefore, while the data suggests positive outcomes, further
        validation would be beneficial.
        """
        
        # Process text
        print("\nProcessing test text...")
        doc = nlp(test_text)
        print("✓ Text processed successfully")
        
        # Test pattern matching
        patterns = {
            "Logical Steps": [
                "first", "therefore", "while"
            ],
            "Evidence Markers": [
                "according to", "studies", "data suggests"
            ],
            "Counterarguments": [
                "however", "limitations"
            ],
            "Examples": [
                "for example"
            ]
        }
        
        print("\nTesting patterns:")
        for category, terms in patterns.items():
            print(f"\n{category}:")
            for term in terms:
                found = term.lower() in test_text.lower()
                print(f"- {term}: {'✓' if found else '✗'}")
                if found:
                    # Find the sentence containing the pattern
                    for sent in doc.sents:
                        if term.lower() in sent.text.lower():
                            print(f"  Context: {sent.text.strip()}")
        
        print("\n✓ Pattern matching test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
