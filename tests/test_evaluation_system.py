"""
Tests for the unified evaluation system.

These tests verify the functionality of the core evaluation system components,
including the MetricRegistry and ModelEvaluator classes.
"""
import pytest
import asyncio
from typing import Dict, Any, List
import os
import sys
from pathlib import Path

# Add the project root to the Python path if not already there
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation_framework.evaluation_system import MetricRegistry, ModelEvaluator, TextSimilarityService # Updated path


class TestMetricRegistry:
    """Tests for the MetricRegistry class."""
    
    def test_registry_initialisation(self, mock_env):
        """Test that the registry initialises properly."""
        registry = MetricRegistry(mock_env)
        assert registry is not None
        assert isinstance(registry.metrics, dict)
    
    def test_registry_discover_metrics(self, mock_env):
        """Test that the registry discovers metrics from the metrics directory."""
        registry = MetricRegistry(mock_env)
        registry.discover_metrics()
        
        # The registry should have discovered at least some metrics
        assert len(registry.metrics) > 0
        
        # Check for some expected core metrics
        expected_metrics = ['accuracy', 'consistency', 'hallucination_detection', 
                           'toxicity', 'bias', 'robustness', 'efficiency']
        
        for metric_name in expected_metrics:
            assert any(metric_name in key for key in registry.metrics.keys()), f"Expected metric '{metric_name}' not found"
    
    def test_registry_get_metric(self, mock_env):
        """Test retrieving metrics from the registry."""
        registry = MetricRegistry(mock_env)
        registry.discover_metrics()
        
        # Get a specific metric
        metric = registry.get_metric('accuracy')
        assert metric is not None
        assert hasattr(metric, 'compute')
        assert hasattr(metric, 'name')
    
    def test_registry_get_nonexistent_metric(self, mock_env):
        """Test that getting a nonexistent metric raises an appropriate error."""
        registry = MetricRegistry(mock_env)
        registry.discover_metrics()
        
        with pytest.raises(KeyError):
            registry.get_metric('nonexistent_metric')


class TestTextSimilarityService:
    """Tests for the TextSimilarityService class."""
    
    def test_service_initialisation(self, mock_env):
        """Test that the service initialises properly."""
        service = TextSimilarityService(mock_env)
        assert service is not None
    
    @pytest.mark.asyncio
    async def test_get_text_embedding(self, mock_env):
        """Test getting text embeddings."""
        service = TextSimilarityService(mock_env)
        embedding = await service.get_embedding("This is a test")
        
        assert embedding is not None
        assert len(embedding) > 0  # Embedding should have a non-zero dimension
    
    @pytest.mark.asyncio
    async def test_calculate_similarity(self, mock_env, text_comparison_cases):
        """Test calculating text similarity."""
        service = TextSimilarityService(mock_env)
        
        for case in text_comparison_cases:
            similarity = await service.calculate_similarity(
                case['generated'], case['reference']
            )
            
            assert 0 <= similarity <= 1, f"Similarity score should be between 0 and 1, got {similarity}"
            
            # Check the similarity aligns with our expectations
            if case['description'] == "Identical sentences":
                assert similarity > 0.95, "Identical sentences should have very high similarity"
            elif case['description'] == "Completely different sentences":
                assert similarity < 0.5, "Different sentences should have low similarity"


class TestModelEvaluator:
    """Tests for the ModelEvaluator class."""
    
    def test_evaluator_initialisation(self, mock_env):
        """Test that the evaluator initialises properly."""
        evaluator = ModelEvaluator(mock_env)
        assert evaluator is not None
        assert evaluator.registry is not None
        assert evaluator.similarity_service is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_model_basic(self, evaluator, sample_test_cases):
        """Test basic model evaluation with default metrics."""
        result = await evaluator.evaluate_model(
            model_id="test-model",
            test_cases=sample_test_cases,
            metrics=['accuracy']
        )
        
        assert result is not None
        assert hasattr(result, 'model_id')
        assert result.model_id == "test-model"
        assert hasattr(result, 'metrics')
        assert 'accuracy' in result.metrics
        assert 0 <= result.metrics['accuracy'] <= 1
    
    @pytest.mark.asyncio
    async def test_evaluate_model_multiple_metrics(self, evaluator, sample_test_cases):
        """Test model evaluation with multiple metrics."""
        metrics = ['accuracy', 'consistency', 'toxicity']
        result = await evaluator.evaluate_model(
            model_id="test-model",
            test_cases=sample_test_cases,
            metrics=metrics
        )
        
        assert result is not None
        assert all(metric in result.metrics for metric in metrics)
    
    @pytest.mark.asyncio
    async def test_calculate_text_metrics(self, evaluator, text_comparison_cases):
        """Test direct calculation of text metrics."""
        case = text_comparison_cases[0]  # Use the first test case
        
        metrics = ['rouge_score', 'bleu_score', 'bert_score']
        result = await evaluator.calculate_text_metrics(
            generated_text=case['generated'],
            reference_text=case['reference'],
            metrics=metrics
        )
        
        assert result is not None
        assert all(metric in result for metric in metrics)
    
    @pytest.mark.asyncio
    async def test_save_evaluation(self, evaluator, sample_test_cases):
        """Test saving evaluation results."""
        result = await evaluator.evaluate_model(
            model_id="test-model",
            test_cases=sample_test_cases,
            metrics=['accuracy']
        )
        
        # Save the result
        result_path = evaluator.save_evaluation(result)
        
        # Verify the file exists
        assert Path(result_path).exists()
        
        # Clean up
        Path(result_path).unlink(missing_ok=True)
