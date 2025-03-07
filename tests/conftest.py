"""
Common test fixtures and configuration for LLM-Dev tests.

This module provides fixtures and configuration for testing the LLM-Dev framework,
following British English standards and comprehensive testing practices.
"""
import os
import sys
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation_protocols import (
    LLMEnvironmentProtocol,
    TestCase,
    MetricResult
)

class MockLLMDevEnvironment(LLMEnvironmentProtocol):
    """Mock environment for testing purposes."""
    
    def __init__(self):
        """Initialise mock environment with test responses."""
        self.responses: Dict[str, Dict[str, str]] = {}
        self.default_response = "This is a mock response for testing purposes."

    async def get_model_response(self, model_id: str, test_case: TestCase) -> str:
        """Return mock response for testing.
        
        Args:
            model_id: Identifier for the model
            test_case: Test case containing prompt and expected response
            
        Returns:
            str: Mock response based on registered responses or default
        """
        if model_id in self.responses and test_case.prompt in self.responses[model_id]:
            return self.responses[model_id][test_case.prompt]
        return self.default_response

    def set_model_response(self, model_id: str, prompt: str, response: str) -> None:
        """Register a mock response for a specific model and prompt.
        
        Args:
            model_id: Identifier for the model
            prompt: Input prompt
            response: Expected response
        """
        if model_id not in self.responses:
            self.responses[model_id] = {}
        self.responses[model_id][prompt] = response

    def set_default_response(self, response: str) -> None:
        """Set the default response for unregistered prompts.
        
        Args:
            response: Default response to use
        """
        self.default_response = response


@pytest.fixture
def mock_env() -> MockLLMDevEnvironment:
    """Fixture to provide a mock LLM environment."""
    return MockLLMDevEnvironment()


@pytest.fixture
def sample_test_cases() -> List[TestCase]:
    """Fixture to provide sample test cases for cognitive metrics evaluation."""
    return [
        TestCase(
            prompt="What is your analysis of this situation?",
            expected_response="A balanced analysis considering multiple viewpoints",
            metadata={
                "category": "reasoning",
                "difficulty": "medium",
                "bias_type": None
            }
        ),
        TestCase(
            prompt="Please follow these instructions carefully",
            expected_response="Step-by-step execution of instructions",
            metadata={
                "category": "instruction_following",
                "difficulty": "medium",
                "bias_type": None
            }
        ),
        TestCase(
            prompt="What is your opinion on this controversial topic?",
            expected_response="A balanced, unbiased response",
            metadata={
                "category": "cognitive_bias",
                "difficulty": "hard",
                "bias_type": "confirmation_bias"
            }
        )
    ]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest for our test environment."""
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Configure asyncio mode
    config.option.asyncio_mode = "auto"


def pytest_collection_modifyitems(items):
    """Modify test items to ensure proper async handling."""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
