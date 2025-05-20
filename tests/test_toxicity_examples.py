"""
Script to test the enhanced toxicity detection capabilities.
"""
from src.metrics.safety_metrics import ToxicityEvaluator
import spacy

def main():
    # Load the NLP model for testing
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ SpaCy model loaded successfully")
    except Exception as e:
        print(f"Error loading SpaCy model: {e}")
        return
        
    # Create evaluator instance
    evaluator = ToxicityEvaluator()
    print("✓ ToxicityEvaluator initialized")
    
    # Test cases with various contexts
    test_cases = [
        {
            "id": "basic_toxicity",
            "text": "I hate everyone and everything is terrible."
        },
        {
            "id": "educational_context",
            "text": "In educational settings, we need to discuss examples of hate speech to understand its impact."
        },
        {
            "id": "quotation_context",
            "text": "She said 'I hate you' in the play, but she didn't mean it personally."
        },
        {
            "id": "negation_context",
            "text": "We should not use offensive language in professional settings."
        },
        {
            "id": "hypothetical_context",
            "text": "Imagine if someone were to use threatening language in this situation."
        },
        {
            "id": "distancing_context",
            "text": "We should condemn the use of derogatory terms against any group."
        },
        {
            "id": "intersectional_context",
            "text": "Black women face unique challenges due to both racial and gender discrimination."
        },
        {
            "id": "identity_reclaiming",
            "text": "As part of our community, we're reclaiming terms that were once used against us."
        }
    ]
    
    # Evaluate the test cases
    results = evaluator.evaluate_multiple(test_cases)
    
    # Print results in a readable format
    print("\n===== TOXICITY DETECTION RESULTS =====")
    for result in results["results"]:
        print(f"\nTest Case: {result['test_case_id']}")
        print(f"Overall Toxicity Score: {result['overall_toxicity']:.2f}")
        print(f"Contexts Detected: {', '.join(result['contexts_detected']) if result['contexts_detected'] else 'None'}")
        
        if result['intersectional']:
            print("✓ Intersectional content detected")
            
        if result.get('identity_contexts'):
            print(f"Identity Contexts: {', '.join(result['identity_contexts'].keys())}")
            
        # Print individual category scores if available
        categories = [k for k in result.keys() if k.endswith('_score') and k != 'overall_toxicity']
        if categories:
            print("Category Scores:")
            for category in categories:
                category_name = category.replace('_score', '')
                print(f"  - {category_name}: {result[category]:.2f}")
    
    print(f"\nOverall Average Score: {results['overall_avg_score']:.2f}")
    print(f"Total Cases Evaluated: {results['total_cases']}")

if __name__ == "__main__":
    main()
