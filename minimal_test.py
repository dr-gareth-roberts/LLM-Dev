"""
Minimal test script for cognitive metrics evaluation.

This script provides a simplified test environment for validating our metrics,
following British English standards and proper type safety.
"""
import asyncio
import pytest
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluation_framework.evaluation_protocols import TestCase, MetricResult, LLMEnvironmentProtocol
from src.metrics.cognitive_metrics import ReasoningEvaluator

class SimpleTestEnvironment(LLMEnvironmentProtocol):
    """Simple test environment for cognitive metrics."""
    
    async def get_model_response(self, model_id: str, test_case: TestCase) -> str:
        """Return a predefined response for testing."""
        return (
            "First, let's examine the evidence. The research indicates positive "
            "outcomes, as demonstrated by multiple studies. However, we must "
            "consider some limitations. For example, the sample size was "
            "relatively small. Therefore, while the approach shows promise, "
            "further validation is needed."
        )

async def test_reasoning_evaluator():
    """Test the reasoning evaluator with a simple test case."""
    # Set up test environment
    env = SimpleTestEnvironment()
    evaluator = ReasoningEvaluator(env)
    
    # Create test case
    test_case = TestCase(
        prompt="Analyse the effectiveness of this approach.",
        expected_response="A reasoned analysis",
        metadata={"category": "reasoning"}
    )
    
    # Run evaluation
    result = await evaluator.evaluate("test_model", [test_case])
    
    # Verify result structure
    assert isinstance(result, MetricResult)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert len(result.details) == 1
    
    # Print results
    print("\nTest Results:")
    print(f"Score: {result.score:.2f}")
    print("\nDetails:")
    for detail in result.details:
        print(f"- Components found: {len(detail['components'])}")
        print(f"- Analysis scores: {detail['analysis']}")

if __name__ == "__main__":
    asyncio.run(test_reasoning_evaluator())
