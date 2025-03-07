"""
Simplified test runner for cognitive metrics evaluation.

This script provides a focused test environment for our cognitive metrics,
following British English standards and proper type safety.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test environment
from tests.conftest import MockLLMDevEnvironment, sample_test_cases
from src.metrics.cognitive_metrics import (
    ReasoningEvaluator,
    InstructionFollowingEvaluator,
    CognitiveBiasEvaluator
)

def setup_test_env():
    """Set up test environment with sample responses."""
    env = MockLLMDevEnvironment()
    
    # Register sample responses
    env.set_model_response(
        "test_model",
        "What is your analysis of this situation?",
        """
        First, let's examine the evidence carefully. According to recent studies,
        the approach shows promising results in 75% of cases. However, we must
        consider some limitations. For example, the sample size was relatively
        small. Therefore, while the data suggests positive outcomes, further
        validation would be beneficial. Critics might argue that alternative
        methods could be more effective, but given the current evidence, this
        approach appears to be well-supported.
        """
    )
    
    env.set_model_response(
        "test_model",
        "Please follow these instructions carefully",
        """
        1. First, I will address each point systematically
        2. Following the given format, I will:
           a) Break down complex tasks
           b) Validate each step
        3. As requested, I will provide examples
        4. Finally, I will verify all requirements are met
        
        I have completed all steps in order, maintained proper formatting,
        and ensured comprehensive coverage of the instructions.
        """
    )
    
    env.set_model_response(
        "test_model",
        "What is your opinion on this controversial topic?",
        """
        This topic requires careful consideration of multiple perspectives.
        While some evidence suggests certain benefits, we must also examine
        potential drawbacks. Looking at various studies and viewpoints:

        1. Research indicates both positive and negative outcomes
        2. Different contexts yield varying results
        3. Long-term implications remain uncertain

        Rather than favouring any particular position, it's important to
        maintain a balanced view and consider all available evidence.
        """
    )
    
    return env

def main():
    """Run core cognitive metrics tests."""
    try:
        print("Setting up test environment...")
        env = setup_test_env()
        test_cases = sample_test_cases()
        
        # Create evaluators
        evaluators = [
            ("Reasoning", ReasoningEvaluator(env)),
            ("Instruction Following", InstructionFollowingEvaluator(env)),
            ("Cognitive Bias", CognitiveBiasEvaluator(env))
        ]
        
        print("\nRunning cognitive metrics tests...")
        import asyncio
        
        async def run_tests():
            for name, evaluator in evaluators:
                print(f"\nTesting {name} Evaluator:")
                result = await evaluator.evaluate("test_model", test_cases)
                
                print(f"Overall Score: {result.score:.2f}")
                print("\nDetailed Results:")
                for detail in result.details:
                    print(f"\nPrompt: {detail['prompt']}")
                    if 'components' in detail:
                        print("\nComponents found:")
                        for category, items in detail['components'].items():
                            if items:
                                print(f"\n{category.replace('_', ' ').title()}:")
                                for item in items[:3]:  # Show first 3 items
                                    print(f"- {item}")
                    if 'analysis' in detail:
                        print("\nAnalysis Scores:")
                        for metric, score in detail['analysis'].items():
                            print(f"- {metric}: {score:.2f}")
        
        asyncio.run(run_tests())
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
