"""
Core test script for cognitive metrics.

This script provides a minimal test environment for our metrics,
following British English standards and proper type safety.
"""
import spacy
from pathlib import Path
import sys

def main():
    """Test core NLP functionality."""
    print("Testing Core NLP Components")
    print("==========================")
    
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
            "Evidence": [
                "according to", "studies", "data suggests"
            ],
            "Counterarguments": [
                "however", "limitations"
            ]
        }
        
        print("\nAnalysing patterns:")
        for category, terms in patterns.items():
            matches = []
            print(f"\n{category}:")
            for term in terms:
                if term.lower() in test_text.lower():
                    matches.append(term)
                    print(f"✓ Found: {term}")
                    # Show context
                    for sent in doc.sents:
                        if term.lower() in sent.text.lower():
                            print(f"  Context: {sent.text.strip()}")
            
            if not matches:
                print(f"✗ No matches found for {category}")
        
        # Test sentence segmentation
        print("\nTesting sentence segmentation:")
        sentences = list(doc.sents)
        print(f"✓ Found {len(sentences)} sentences")
        
        # Test dependency parsing
        print("\nTesting dependency parsing:")
        for token in doc[:5]:
            print(f"Token: {token.text}")
            print(f"Dependency: {token.dep_}")
            print(f"Head: {token.head.text}")
            print()
        
        print("\n✓ All core tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
