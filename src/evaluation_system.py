"""
evaluation_system.py - Unified Evaluation System

This module implements a consolidated evaluation system with a plugin architecture,
following the plan to standardise code organisation and create a more modular system.
"""
import inspect
import json
import pkgutil
from dataclasses import dataclass, asdict
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation_protocols import (
    BaseMetricEvaluator,
    BaseModelEvaluator,
    EvaluationMetadata,
    EvaluationResultProtocol,
    LLMEnvironmentProtocol,
    MetricResult,
    TestCase,
    TextSimilarityEvaluator,
)


@dataclass
class EvaluationResult(EvaluationResultProtocol):
    """Dataclass for storing evaluation results."""
    model_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    detailed_results: Dict[str, Any]
    metadata: Dict[str, Any]
    error_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation result to a dictionary."""
        result_dict = asdict(self)
        result_dict["timestamp"] = self.timestamp.isoformat()
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create an EvaluationResult from a dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class MetricRegistry:
    """Registry for metric evaluators."""
    
    _registry: Dict[str, Type[BaseMetricEvaluator]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseMetricEvaluator]], Type[BaseMetricEvaluator]]:
        """Decorator to register a metric evaluator."""
        def decorator(evaluator_class: Type[BaseMetricEvaluator]) -> Type[BaseMetricEvaluator]:
            cls._registry[name] = evaluator_class
            return evaluator_class
        return decorator
    
    @classmethod
    def get_evaluator(cls, name: str) -> Optional[Type[BaseMetricEvaluator]]:
        """Get a metric evaluator by name."""
        return cls._registry.get(name)
    
    @classmethod
    def get_all_metrics(cls) -> List[str]:
        """Get all registered metric names."""
        return list(cls._registry.keys())
    
    @classmethod
    def discover_metrics(cls, package_name: str = "src.metrics") -> None:
        """Discover and register metrics from a package."""
        try:
            package = import_module(package_name)
            for _, modname, ispkg in pkgutil.iter_modules(
                package.__path__, package.__name__ + "."
            ):
                if ispkg:
                    cls.discover_metrics(modname)
                else:
                    import_module(modname)
        except (ImportError, AttributeError) as e:
            # Package doesn't exist or has no __path__ attribute
            pass


class TextSimilarityService(TextSimilarityEvaluator):
    """Service for text similarity evaluation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.85):
        self._model_name = model_name
        self._model = None
        self._similarity_threshold = similarity_threshold
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using sentence transformer."""
        return self.model.encode([text.strip().lower()])[0]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        # Get embeddings
        embedding1 = self.get_text_embedding(text1)
        embedding2 = self.get_text_embedding(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def compare_responses(self, actual: str, expected: str, threshold: Optional[float] = None) -> bool:
        """Compare if two responses are semantically similar."""
        if not actual or not expected:
            return False
        
        threshold = threshold or self._similarity_threshold
        similarity = self.calculate_similarity(actual, expected)
        
        return similarity >= threshold


class ModelEvaluator(BaseModelEvaluator):
    """Model evaluator implementation."""
    
    def __init__(self, env: LLMEnvironmentProtocol):
        self.env = env
        self.results_dir = env.env_config.output_dir / 'evaluations'
        self.results_dir.mkdir(exist_ok=True)
        self.current_evaluation: Optional[EvaluationResult] = None
        self.similarity_service = TextSimilarityService()
        
        # Discover available metrics
        MetricRegistry.discover_metrics()
    
    async def evaluate_model(self, 
                           model_id: str,
                           test_cases: List[TestCase],
                           metrics: Optional[List[str]] = None) -> EvaluationResult:
        """
        Comprehensive model evaluation across multiple dimensions.
        """
        if metrics is None:
            metrics = MetricRegistry.get_all_metrics()
        
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'metrics': {},
            'detailed_results': {},
            'error_analysis': {},
            'metadata': {
                'test_cases': len(test_cases), 
                'metrics': metrics,
                'additional_info': {'evaluation_version': '2.0'}
            }
        }
        
        try:
            # Run evaluations
            for metric in metrics:
                metric_result = await self.evaluate_metric(model_id, test_cases, metric)
                results['metrics'][metric] = metric_result['value']
                results['detailed_results'][metric] = metric_result['details']
            
            # Error analysis
            results['error_analysis'] = await self._perform_error_analysis(
                model_id, test_cases, results['detailed_results']
            )
            
            # Save results
            self.current_evaluation = EvaluationResult(**results)
            self.save_evaluation(self.current_evaluation)
            
            return self.current_evaluation
            
        except Exception as e:
            self.env.logger.error(f"Evaluation error for model {model_id}: {str(e)}")
            raise
    
    async def evaluate_metric(self, 
                            model_id: str,
                            test_cases: List[TestCase],
                            metric: str) -> MetricResult:
        """Evaluate a specific metric."""
        evaluator_class = MetricRegistry.get_evaluator(metric)
        
        if evaluator_class:
            evaluator = evaluator_class(self.env, self.similarity_service)
            return await evaluator.evaluate(model_id, test_cases)
        else:
            self.env.logger.warning(f"Unsupported metric: {metric}")
            return {
                "value": 0.0,
                "details": {
                    "error": f"Unsupported metric: {metric}",
                }
            }
    
    async def _perform_error_analysis(self, 
                                    model_id: str, 
                                    test_cases: List[TestCase],
                                    detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform error analysis on evaluation results."""
        error_analysis = {
            "error_cases": [],
            "error_patterns": {},
            "metrics_correlation": {},
        }
        
        # Extract error cases across metrics
        for metric, details in detailed_results.items():
            if "case_details" in details:
                for i, case_detail in enumerate(details["case_details"]):
                    if case_detail.get("correct") is False:
                        error_analysis["error_cases"].append({
                            "metric": metric,
                            "test_case_index": i,
                            "input": case_detail.get("input", ""),
                            "expected": case_detail.get("expected", ""),
                            "actual": case_detail.get("actual", ""),
                            "error_type": self._determine_error_type(
                                case_detail.get("expected", ""), 
                                case_detail.get("actual", "")
                            )
                        })
        
        # Analyze error patterns
        error_types = [case["error_type"] for case in error_analysis["error_cases"]]
        for error_type in set(error_types):
            error_analysis["error_patterns"][error_type] = error_types.count(error_type)
        
        return error_analysis
    
    def _determine_error_type(self, expected: str, actual: str) -> str:
        """Determine the type of error between expected and actual outputs."""
        if not actual:
            return "no_response"
        
        similarity = self.similarity_service.calculate_similarity(expected, actual)
        
        if similarity > 0.7:
            return "minor_deviation"
        elif similarity > 0.4:
            return "partial_match"
        else:
            return "complete_mismatch"
    
    def save_evaluation(self, result: EvaluationResult, file_path: Optional[Path] = None) -> Path:
        """Save evaluation results to a file."""
        if file_path is None:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            file_path = self.results_dir / f"evaluation_{result.model_id}_{timestamp}.json"
        
        with open(file_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return file_path
"""
