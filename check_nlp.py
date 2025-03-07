"""
NLP validation script.
Validates core functionality required by the metrics system.
"""
import sys

def validate_nlp():
    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded")
        
        doc = nlp("Testing NLP.")
        print(f"Processed {len(list(doc.sents))} sentences")
        return True
    except ImportError:
        print("Error: Run 'pip install spacy'")
        return False
    except OSError:
        print("Error: Run 'python -m spacy download en_core_web_sm'")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if not validate_nlp():
        sys.exit(1)
    print("Success!")
