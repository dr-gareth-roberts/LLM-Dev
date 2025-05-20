"""
Tests for cognitive evaluation metrics.

These tests verify the functionality of the cognitive metrics evaluators,
following British English standards and comprehensive testing practices.
"""
import pytest
import asyncio
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.cognitive_metrics import (
    ReasoningEvaluator,
    InstructionFollowingEvaluator,
    CognitiveBiasEvaluator
)
from src.evaluation_framework.evaluation_protocols import ( # Updated path
    TestCase,
    MetricResult,
    MetricCategory
)


@pytest.mark.asyncio
class TestReasoningEvaluator:
    """Tests for the ReasoningEvaluator class."""

    @pytest.fixture
    async def evaluator(self, mock_env):
        """Create a ReasoningEvaluator instance."""
        return ReasoningEvaluator(mock_env)

    @pytest.mark.parametrize("test_input,expected_components", [
        (
            "First, we must consider the evidence. Therefore, based on the data, "
            "we can conclude that this approach is optimal. However, we should "
            "also consider alternative viewpoints.",
            {
                "logical_steps": ["First", "Therefore", "conclude"],
                "evidence": ["based on the data"],
                "assumptions": [],
                "counterarguments": ["However", "alternative viewpoints"]
            }
        ),
        (
            "Let's assume the system is stable. The research indicates positive "
            "outcomes, although some limitations exist. For example, the sample "
            "size was limited.",
            {
                "logical_steps": [],
                "evidence": ["research indicates", "For example"],
                "assumptions": ["assume"],
                "counterarguments": ["although", "limitations"]
            }
        )
    ])
    async def test_extract_reasoning_components(
        self,
        evaluator,
        test_input,
        expected_components
    ):
        """Test extraction of reasoning components."""
        components = evaluator._extract_reasoning_components(test_input)
        
        for component_type, expected in expected_components.items():
            assert component_type in components
            for marker in expected:
                assert any(
                    marker.lower() in found.lower()
                    for found in components[component_type]
                )

    @pytest.mark.parametrize("components,expected_scores", [
        (
            {
                "logical_steps": ["First", "Therefore", "conclude"],
                "evidence": ["based on the data"],
                "assumptions": [],
                "counterarguments": ["However", "alternative viewpoints"]
            },
            {
                "logical_coherence": 0.7,
                "evidence_quality": 0.33,
                "assumption_awareness": 0.0,
                "counterargument_handling": 0.8
            }
        ),
        (
            {
                "logical_steps": [],
                "evidence": ["research indicates", "For example"],
                "assumptions": ["assume"],
                "counterarguments": ["although", "limitations"]
            },
            {
                "logical_coherence": 0.2,
                "evidence_quality": 0.67,
                "assumption_awareness": 0.33,
                "counterargument_handling": 0.5
            }
        )
    ])
    async def test_analyse_reasoning_quality(
        self,
        evaluator,
        components,
        expected_scores
    ):
        """Test analysis of reasoning quality."""
        analysis = evaluator._analyse_reasoning_quality(components)
        
        for aspect, expected_score in expected_scores.items():
            assert aspect in analysis
            assert abs(analysis[aspect] - expected_score) < 0.1

    async def test_evaluate(self, evaluator, mock_env):
        """Test complete evaluation process."""
        test_cases = [
            TestCase(
                prompt="Analyse the effectiveness of this approach.",
                expected_response="A reasoned analysis",
                metadata={"category": "reasoning"}
            )
        ]

        mock_env.set_model_response(
            "test_model",
            "Analyse the effectiveness of this approach.",
            "First, let's examine the evidence. The research indicates positive "
            "outcomes, as demonstrated by multiple studies. However, we must "
            "consider some limitations. For example, the sample size was "
            "relatively small. Therefore, while the approach shows promise, "
            "further validation is needed."
        )

        result = await evaluator.evaluate("test_model", test_cases)

        assert isinstance(result, MetricResult)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert len(result.details) == len(test_cases)

        detail = result.details[0]
        assert "prompt" in detail
        assert "response" in detail
        assert "components" in detail
        assert "analysis" in detail
        assert "score" in detail


@pytest.mark.asyncio
class TestInstructionFollowingEvaluator:
    """Tests for the InstructionFollowingEvaluator class."""

    @pytest.fixture
    async def evaluator(self, mock_env):
        """Create an InstructionFollowingEvaluator instance."""
        return InstructionFollowingEvaluator(mock_env)

    @pytest.mark.parametrize("test_input,expected_instructions", [
        (
            "Please analyse the data and list three key findings. "
            "You must provide evidence for each point.",
            {
                "tasks": ["analyse", "list", "provide evidence"],
                "constraints": ["three key findings"],
                "conditions": []
            }
        ),
        (
            "Compare the approaches when the data is available. "
            "Present your analysis in bullet points, excluding any subjective opinions.",
            {
                "tasks": ["Compare", "Present"],
                "constraints": ["in bullet points", "excluding subjective opinions"],
                "conditions": ["when the data is available"]
            }
        )
    ])
    async def test_extract_instructions(
        self,
        evaluator,
        test_input,
        expected_instructions
    ):
        """Test extraction of instructions from prompts."""
        instructions = evaluator._extract_instructions(test_input)
        
        for category, expected in expected_instructions.items():
            assert category in instructions
            for instruction in expected:
                assert any(
                    instruction.lower() in found.lower()
                    for found in instructions[category]
                )

    @pytest.mark.parametrize("instructions,response,expected_scores", [
        (
            {
                "tasks": ["analyse", "list", "provide evidence"],
                "constraints": ["three key findings"],
                "conditions": []
            },
            "Analysis shows three main findings: 1) Growth increased by 25%, "
            "supported by quarterly data. 2) Efficiency improved 15%, as shown "
            "in performance metrics. 3) User satisfaction rose 30%, based on "
            "survey results.",
            {
                "task_completion": 1.0,
                "constraint_adherence": 1.0,
                "condition_satisfaction": 1.0
            }
        ),
        (
            {
                "tasks": ["Compare", "Present"],
                "constraints": ["in bullet points", "excluding subjective opinions"],
                "conditions": ["when the data is available"]
            },
            "When examining the available data:\n- Method A showed 15% better "
            "performance\n- Method B required 30% less resources\n- Both methods "
            "demonstrated similar reliability metrics",
            {
                "task_completion": 1.0,
                "constraint_adherence": 0.5,
                "condition_satisfaction": 1.0
            }
        )
    ])
    async def test_analyse_task_completion(
        self,
        evaluator,
        instructions,
        response,
        expected_scores
    ):
        """Test analysis of task completion."""
        analysis = evaluator._analyse_task_completion(instructions, response)
        
        for aspect, expected_score in expected_scores.items():
            assert aspect in analysis
            assert abs(analysis[aspect] - expected_score) < 0.1

    async def test_evaluate(self, evaluator, mock_env):
        """Test complete evaluation process."""
        test_cases = [
            TestCase(
                prompt="Analyse the data and list three key findings with evidence.",
                expected_response="A structured response",
                metadata={"category": "instruction_following"}
            )
        ]

        mock_env.set_model_response(
            "test_model",
            "Analyse the data and list three key findings with evidence.",
            "Analysis reveals three key findings:\n1. Growth increased 25% "
            "(supported by quarterly data)\n2. Efficiency improved 15% "
            "(shown in performance metrics)\n3. User satisfaction rose 30% "
            "(based on survey results)"
        )

        result = await evaluator.evaluate("test_model", test_cases)

        assert isinstance(result, MetricResult)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert len(result.details) == len(test_cases)

        detail = result.details[0]
        assert "prompt" in detail
        assert "response" in detail
        assert "instructions" in detail
        assert "analysis" in detail
        assert "score" in detail


@pytest.mark.asyncio
class TestCognitiveBiasEvaluator:
    """Tests for the CognitiveBiasEvaluator class."""

    @pytest.fixture
    async def evaluator(self, mock_env):
        """Create a CognitiveBiasEvaluator instance."""
        return CognitiveBiasEvaluator(mock_env)

    @pytest.mark.parametrize("test_input,expected_indicators", [
        (
            "We must focus solely on supporting evidence that proves our point. "
            "This confirms our initial assumptions.",
            {
                "confirmation_bias": ["focus solely", "proves", "confirms"],
                "anchoring_bias": ["initial"],
                "availability_bias": [],
                "logical_fallacies": []
            }
        ),
        (
            "Recent cases clearly show this trend. These notable examples prove "
            "that everyone thinks this way.",
            {
                "confirmation_bias": ["clearly", "prove"],
                "anchoring_bias": [],
                "availability_bias": ["Recent cases", "notable examples"],
                "logical_fallacies": ["everyone thinks"]
            }
        )
    ])
    async def test_extract_bias_indicators(
        self,
        evaluator,
        test_input,
        expected_indicators
    ):
        """Test extraction of cognitive bias indicators."""
        indicators = evaluator._extract_bias_indicators(test_input)
        
        for bias_type in expected_indicators:
            assert bias_type in indicators
            for indicator in expected_indicators[bias_type]:
                assert any(
                    indicator.lower() in found.lower()
                    for found in indicators[bias_type]
                )

    @pytest.mark.parametrize("indicators,expected_scores", [
        (
            {
                "confirmation_bias": ["focus solely", "proves", "confirms"],
                "anchoring_bias": ["initial"],
                "availability_bias": [],
                "logical_fallacies": []
            },
            {
                "confirmation_bias": 0.5,
                "anchoring_bias": 0.25,
                "availability_bias": 0.0,
                "logical_fallacies": 0.0
            }
        ),
        (
            {
                "confirmation_bias": ["clearly", "prove"],
                "anchoring_bias": [],
                "availability_bias": ["Recent cases", "notable examples"],
                "logical_fallacies": ["everyone thinks"]
            },
            {
                "confirmation_bias": 0.25,
                "anchoring_bias": 0.0,
                "availability_bias": 0.5,
                "logical_fallacies": 0.17
            }
        )
    ])
    async def test_analyse_bias_presence(
        self,
        evaluator,
        indicators,
        expected_scores
    ):
        """Test analysis of cognitive bias presence."""
        analysis = evaluator._analyse_bias_presence(indicators)
        
        for bias_type, expected_score in expected_scores.items():
            assert bias_type in analysis
            assert abs(analysis[bias_type] - expected_score) < 0.1

    async def test_evaluate(self, evaluator, mock_env):
        """Test complete evaluation process."""
        test_cases = [
            TestCase(
                prompt="What is your analysis of this situation?",
                expected_response="An unbiased analysis",
                metadata={"category": "cognitive_bias"}
            )
        ]

        mock_env.set_model_response(
            "test_model",
            "What is your analysis of this situation?",
            "Recent cases clearly show this trend. These notable examples prove "
            "that everyone thinks this way. We must focus solely on supporting "
            "evidence that confirms our initial assumptions."
        )

        result = await evaluator.evaluate("test_model", test_cases)

        assert isinstance(result, MetricResult)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert len(result.details) == len(test_cases)

        detail = result.details[0]
        assert "prompt" in detail
        assert "response" in detail
        assert "indicators" in detail
        assert "analysis" in detail
        assert "score" in detail
