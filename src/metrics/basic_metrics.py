"""
basic_metrics.py - Implementation of basic evaluation metrics

This module contains the core evaluation metrics like accuracy and consistency,
following the plugin architecture pattern.
"""
from typing import Any, Dict, List

from src.evaluation_framework.evaluation_protocols import ( # Updated path
    BaseMetricEvaluator,
    LLMEnvironmentProtocol,
    MetricResult,
    TestCase,
    TextSimilarityEvaluator, # Still from here, though its definition is unconfirmed
)
from src.core.metric_registry import MetricRegistry # Corrected path for MetricRegistry


@MetricRegistry.register("accuracy")
class AccuracyEvaluator(BaseMetricEvaluator):
    """Evaluator for model accuracy."""
    
    def __init__(self, env: LLMEnvironmentProtocol, similarity_service: TextSimilarityEvaluator):
        self.env = env
        self.similarity_service = similarity_service
    
    async def evaluate(self, model_id: str, test_cases: List[TestCase]) -> MetricResult:
        """Evaluate model accuracy on test cases."""
        correct = 0
        details = []
        
        for case in test_cases:
            expected = case.get('expected_output', '')
            actual = await self.env.get_model_response(model_id, case['input'])
            
            # Compare responses
            is_correct = self.similarity_service.compare_responses(actual, expected)
            correct += int(is_correct)
            
            details.append({
                'input': case['input'],
                'expected': expected,
                'actual': actual,
                'correct': is_correct,
                'similarity_score': self.similarity_service.calculate_similarity(actual, expected)
            })
        
        accuracy = correct / len(test_cases) if test_cases else 0
        return {
            'value': accuracy,
            'details': {
                'total_cases': len(test_cases),
                'correct_cases': correct,
                'case_details': details
            }
        }


@MetricRegistry.register("consistency")
class ConsistencyEvaluator(BaseMetricEvaluator):
    """Evaluator for model consistency."""
    
    def __init__(self, env: LLMEnvironmentProtocol, similarity_service: TextSimilarityEvaluator):
        self.env = env
        self.similarity_service = similarity_service
    
    async def evaluate(self, model_id: str, test_cases: List[TestCase]) -> MetricResult:
        """Evaluate model consistency by testing similar inputs."""
        consistency_scores = []
        details = []
        
        for case in test_cases:
            variations = case.get('variations', [case['input']])
            if len(variations) < 2:
                continue
                
            # Get responses for each variation
            responses = []
            for variation in variations:
                response = await self.env.get_model_response(model_id, variation)
                responses.append(response)
            
            # Calculate pairwise similarity
            case_consistency = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarity = self.similarity_service.calculate_similarity(
                        responses[i], responses[j]
                    )
                    case_consistency.append(similarity)
            
            avg_consistency = sum(case_consistency) / len(case_consistency) if case_consistency else 0
            consistency_scores.append(avg_consistency)
            
            details.append({
                'input_variations': variations,
                'responses': responses,
                'consistency_score': avg_consistency
            })
        
        overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        return {
            'value': overall_consistency,
            'details': {
                'case_details': details,
                'overall_score': overall_consistency
            }
        }
"""
