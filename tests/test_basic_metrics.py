"""
Tests for basic evaluation metrics.

These tests verify the functionality of the accuracy and consistency metrics.
"""
import pytest
import asyncio
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.basic_metrics import AccuracyMetric, ConsistencyMetric


class TestAccuracyMetric:
    """Tests for the AccuracyMetric class."""
    
    @pytest.mark.asyncio
    async def test_perfect_match(self, mock_env):
        """Test perfect match between expected and actual outputs."""
        metric = AccuracyMetric(mock_env)
        
        test_case = {
            "input": "What is the capital of France?",
            "expected_output": "Paris",
            "model_response": "Paris",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] == 1.0
    
    @pytest.mark.asyncio
    async def test_semantic_match(self, mock_env):
        """Test semantic similarity matching."""
        metric = AccuracyMetric(mock_env)
        
        test_case = {
            "input": "What is the capital of France?",
            "expected_output": "The capital of France is Paris.",
            "model_response": "Paris is the capital city of France.",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] > 0.8  # Should have high semantic similarity
    
    @pytest.mark.asyncio
    async def test_complete_mismatch(self, mock_env):
        """Test completely incorrect response."""
        metric = AccuracyMetric(mock_env)
        
        test_case = {
            "input": "What is the capital of France?",
            "expected_output": "Paris",
            "model_response": "London",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] < 0.5  # Should have low similarity
    
    @pytest.mark.asyncio
    async def test_empty_response(self, mock_env):
        """Test empty model response."""
        metric = AccuracyMetric(mock_env)
        
        test_case = {
            "input": "What is the capital of France?",
            "expected_output": "Paris",
            "model_response": "",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] == 0.0  # Empty response should score 0


class TestConsistencyMetric:
    """Tests for the ConsistencyMetric class."""
    
    @pytest.mark.asyncio
    async def test_consistent_responses(self, mock_env):
        """Test consistent responses to similar inputs."""
        metric = ConsistencyMetric(mock_env)
        
        # Set up mock responses for variations
        mock_env.set_model_response("test-model", 
                               "What is the capital of France?", 
                               "The capital of France is Paris.")
        mock_env.set_model_response("test-model",
                               "Tell me the capital of France", 
                               "Paris is the capital of France.")
        mock_env.set_model_response("test-model",
                               "France's capital city is?", 
                               "Paris is the capital city of France.")
        
        test_case = {
            "input": "What is the capital of France?",
            "variations": [
                "What is the capital of France?",
                "Tell me the capital of France",
                "France's capital city is?"
            ],
            "model_id": "test-model",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] > 0.8  # Should have high consistency
    
    @pytest.mark.asyncio
    async def test_inconsistent_responses(self, mock_env):
        """Test inconsistent responses to similar inputs."""
        metric = ConsistencyMetric(mock_env)
        
        # Set up mock responses for variations
        mock_env.set_model_response("test-model", 
                               "What is the capital of France?", 
                               "The capital of France is Paris.")
        mock_env.set_model_response("test-model",
                               "Tell me the capital of France", 
                               "I believe it's Nice, but I'm not entirely sure.")
        mock_env.set_model_response("test-model",
                               "France's capital city is?", 
                               "Lyon is a major city in France.")
        
        test_case = {
            "input": "What is the capital of France?",
            "variations": [
                "What is the capital of France?",
                "Tell me the capital of France",
                "France's capital city is?"
            ],
            "model_id": "test-model",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] < 0.6  # Should have low consistency
    
    @pytest.mark.asyncio
    async def test_no_variations(self, mock_env):
        """Test case with no variations."""
        metric = ConsistencyMetric(mock_env)
        
        test_case = {
            "input": "What is the capital of France?",
            "model_id": "test-model",
            "category": "factual"
        }
        
        result = await metric.compute(test_case)
        
        assert result is not None
        assert "score" in result
        assert result["score"] == 1.0  # No variations should default to perfect consistency
