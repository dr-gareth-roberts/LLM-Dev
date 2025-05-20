"""
evaluation_protocols.py - Protocol definitions for the evaluation system

This module contains Protocol classes that define the interfaces for the evaluation
system components, following British English standards and comprehensive type safety.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Dict, List, Literal, Optional, Protocol, 
    TypedDict, Union, runtime_checkable, TypeVar, Generic
)


class MetricCategory(Enum):
    """Enumeration of metric categories."""
    FACTUALITY = auto()
    SAFETY = auto()
    PERFORMANCE = auto()
    COGNITIVE = auto()
    CREATIVITY = auto()


class DifficultyLevel(Enum):
    """Enumeration of test case difficulty levels."""
    BASIC = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    EXPERT = auto()


class TestCase:
    """Structure for test cases with comprehensive type safety."""
    
    def __init__(
        self,
        prompt: str,
        expected_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialise a test case.
        
        Args:
            prompt: Input prompt for the model
            expected_response: Expected model response
            metadata: Optional metadata about the test case
        """
        self.prompt = prompt
        self.expected_response = expected_response
        self.metadata = metadata or {}
        self.creation_time = datetime.now()


class MetricResult:
    """Structure for metric evaluation results with comprehensive type safety."""
    
    def __init__(
        self,
        score: float,
        details: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialise a metric result.
        
        Args:
            score: Evaluation score between 0 and 1
            details: List of detailed evaluation results
            metadata: Optional metadata about the evaluation
        """
        if not 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1")
        
        self.score = score
        self.details = details
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


@runtime_checkable
class LLMEnvironmentProtocol(Protocol):
    """Protocol defining the interface for the LLM development environment."""
    
    async def get_model_response(
        self,
        model_id: str,
        test_case: TestCase
    ) -> str:
        """Get a response from a specific model.
        
        Args:
            model_id: Identifier for the model
            test_case: Test case containing prompt and expected response
            
        Returns:
            str: Model's response to the prompt
        """
        ...


class BaseMetricEvaluator(ABC):
    """Abstract base class for metric evaluators."""
    
    def __init__(self, env: LLMEnvironmentProtocol):
        """Initialise the metric evaluator.
        
        Args:
            env: LLM environment for model interaction
        """
        self.env = env
    
    @abstractmethod
    async def evaluate(
        self,
        model_id: str,
        test_cases: List[TestCase]
    ) -> MetricResult:
        """Evaluate a specific metric on the given test cases.
        
        Args:
            model_id: Identifier for the model to evaluate
            test_cases: List of test cases to evaluate
            
        Returns:
            MetricResult: Evaluation results with score and details
        """
        pass


class BaseModelEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    @abstractmethod
    async def evaluate_model(
        self, 
        model_id: str,
        test_cases: List[TestCase],
        metrics: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, MetricResult]:
        """Evaluate a model on the given test cases using specified metrics.
        
        Args:
            model_id: Identifier for the model to evaluate
            test_cases: List of test cases to evaluate
            metrics: Optional list of specific metrics to evaluate
            batch_size: Optional batch size for evaluation
            
        Returns:
            Dict[str, MetricResult]: Mapping of metric names to their results
        """
        pass
    
    @abstractmethod
    async def evaluate_metric(
        self, 
        model_id: str,
        test_cases: List[TestCase],
        metric: str
    ) -> MetricResult:
        """Evaluate a specific metric for a model.
        
        Args:
            model_id: Identifier for the model to evaluate
            test_cases: List of test cases to evaluate
            metric: Name of the metric to evaluate
            
        Returns:
            MetricResult: Evaluation results for the specified metric
        """
        pass
    
    @abstractmethod
    def save_evaluation(
        self, 
        results: Dict[str, MetricResult], 
        file_path: Optional[Path] = None,
        format: Literal["json", "yaml", "csv"] = "json"
    ) -> Path:
        """Save evaluation results to a file.
        
        Args:
            results: Mapping of metric names to their results
            file_path: Optional path to save results
            format: Format to save results in
            
        Returns:
            Path: Path where results were saved
        """
        pass
    
    @abstractmethod
    async def batch_evaluate(
        self,
        model_ids: List[str],
        test_cases: List[TestCase],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, MetricResult]]:
        """Evaluate multiple models in parallel.
        
        Args:
            model_ids: List of model identifiers
            test_cases: List of test cases to evaluate
            metrics: Optional list of specific metrics to evaluate
            
        Returns:
            Dict[str, Dict[str, MetricResult]]: Mapping of model IDs to their metric results
        """
        pass
