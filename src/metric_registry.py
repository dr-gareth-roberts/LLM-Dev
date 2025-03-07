"""
metric_registry.py - Central registry for evaluation metrics

This module implements a plugin-based registry system for managing evaluation metrics,
following the plan to standardise code organisation and implement consistent interfaces.
"""
from typing import Dict, Type, List, Optional
from src.evaluation_protocols import (
    BaseMetricEvaluator,
    MetricCategory,
    LLMEnvironmentProtocol
)


class MetricRegistry:
    """Central registry for managing evaluation metrics."""
    
    _metrics: Dict[str, Type[BaseMetricEvaluator]] = {}
    _categories: Dict[str, MetricCategory] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        category: MetricCategory,
        override: bool = False
    ):
        """Decorator to register a metric evaluator.
        
        Args:
            name: Name of the metric
            category: Category of the metric
            override: Whether to override existing registration
        """
        def decorator(metric_class: Type[BaseMetricEvaluator]) -> Type[BaseMetricEvaluator]:
            if name in cls._metrics and not override:
                raise ValueError(f"Metric '{name}' is already registered")
                
            cls._metrics[name] = metric_class
            cls._categories[name] = category
            return metric_class
            
        return decorator
    
    @classmethod
    def get_metric(
        cls,
        name: str,
        env: LLMEnvironmentProtocol
    ) -> BaseMetricEvaluator:
        """Get a metric evaluator instance by name.
        
        Args:
            name: Name of the metric
            env: LLM environment instance
            
        Returns:
            Instance of the metric evaluator
            
        Raises:
            KeyError: If metric is not registered
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
            
        metric_class = cls._metrics[name]
        return metric_class(env)
    
    @classmethod
    def list_metrics(
        cls,
        category: Optional[MetricCategory] = None
    ) -> List[str]:
        """List registered metrics, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of metric names
        """
        if category:
            return [
                name for name, cat in cls._categories.items()
                if cat == category
            ]
        return list(cls._metrics.keys())
    
    @classmethod
    def get_category(cls, name: str) -> MetricCategory:
        """Get the category of a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Category of the metric
            
        Raises:
            KeyError: If metric is not registered
        """
        if name not in cls._categories:
            raise KeyError(f"Metric '{name}' is not registered")
            
        return cls._categories[name]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics."""
        cls._metrics.clear()
        cls._categories.clear()
    
    @classmethod
    def register_batch(
        cls,
        metrics: Dict[str, Type[BaseMetricEvaluator]],
        categories: Dict[str, MetricCategory],
        override: bool = False
    ) -> None:
        """Register multiple metrics at once.
        
        Args:
            metrics: Dictionary mapping names to metric classes
            categories: Dictionary mapping names to categories
            override: Whether to override existing registrations
        """
        for name, metric_class in metrics.items():
            if name not in categories:
                raise ValueError(f"Category not provided for metric '{name}'")
                
            if name in cls._metrics and not override:
                raise ValueError(f"Metric '{name}' is already registered")
                
            cls._metrics[name] = metric_class
            cls._categories[name] = categories[name]
