"""
Test script for ReasoningEvaluator functionality.

This script validates the core functionality of our reasoning evaluation system,
following British English standards and proper type safety.
"""
import asyncio
import pytest
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluation_protocols import TestCase, MetricResult, LLMEnvironmentProtocol
from src.metrics.cognitive_metrics import ReasoningEvaluator

class MockLLMEnvironment(LLMEnvironmentProtocol):
    """Mock environment for testing reasoning evaluation."""
    
    async def get_model_response(self, model_id: str, test_case: TestCase) -> str:
        """Return a predefined response with clear reasoning patterns."""
        return (
            "First, let's examine the evidence. According to recent studies, "
            "the approach shows promising results in 75% of cases. However, "
            "we must consider some limitations. For example, the sample size "
            "was relatively small. Therefore, while the data suggests positive "
            "outcomes, further validation would be beneficial. Critics might argue "
            "that alternative methods could be more effective, but given the "
            "current evidence, this approach appears to be well-supported."
        )

async def test_reasoning_components():
    """Test reasoning component extraction and analysis."""
    # Set up test environment
    env = MockLLMEnvironment()
    evaluator = ReasoningEvaluator(env)
    
    # Create test case
    test_case = TestCase(
        prompt="Analyse the effectiveness of this approach based on available evidence.",
        expected_response="A well-reasoned analysis with evidence and counterarguments",
        metadata={"category": "reasoning", "complexity": "high"}
    )
    
    # Run evaluation
    result = await evaluator.evaluate("test_model", [test_case])
    
    # Validate result structure
    assert isinstance(result, MetricResult)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert len(result.details) == 1
    
    # Validate components
    detail = result.details[0]
    components = detail["components"]
    
    # Check for logical steps
    assert len(components["logical_steps"]) > 0
    assert any("first" in step.lower() for step in components["logical_steps"])
    assert any("therefore" in step.lower() for step in components["logical_steps"])
    
    # Check for evidence
    assert len(components["evidence"]) > 0
    assert any("according to" in ev.lower() for ev in components["evidence"])
    assert any("for example" in ev.lower() for ev in components["evidence"])
    
    # Check for counterarguments
    assert len(components["counterarguments"]) > 0
    assert any("however" in arg.lower() for arg in components["counterarguments"])
    assert any("critics might argue" in arg.lower() for arg in components["counterarguments"])
    
    # Print results
    print("\nReasoning Evaluation Results:")
    print(f"Overall Score: {result.score:.2f}")
    print("\nComponent Analysis:")
    for category, items in components.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"- {item}")
    
    print("\nAnalysis Scores:")
    for metric, score in detail["analysis"].items():
        print(f"- {metric}: {score:.2f}")

if __name__ == "__main__":
    asyncio.run(test_reasoning_components())
