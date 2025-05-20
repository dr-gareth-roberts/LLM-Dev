"""
LLM-Dev Framework - A comprehensive framework for LLM development and evaluation.

This package provides tools and metrics for evaluating language models across multiple
dimensions including factuality, safety, cognitive capabilities, and performance.
"""

from .evaluation_framework.evaluation_protocols import (
    LLMEnvironmentProtocol,
    BaseMetricEvaluator,
    TestCase,
    MetricResult,
    MetricCategory
)

from .metrics import (
    ReasoningEvaluator,
    InstructionFollowingEvaluator,
    CognitiveBiasEvaluator
)

__version__ = "0.1.0"
__author__ = "Dr. Gareth Roberts"

__all__ = [
    "LLMEnvironmentProtocol",
    "BaseMetricEvaluator",
    "TestCase",
    "MetricResult",
    "MetricCategory",
    "ReasoningEvaluator",
    "InstructionFollowingEvaluator",
    "CognitiveBiasEvaluator"
]
