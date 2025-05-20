"""
creative_metrics.py - Implementation of creativity evaluation metrics

This module contains evaluators for assessing model creativity and originality,
following the plugin architecture pattern.
"""
import re
import statistics
from typing import Any, Dict, List, Set, Tuple

from src.evaluation_framework.evaluation_protocols import (
    BaseMetricEvaluator,
    LLMEnvironmentProtocol,
    MetricResult,
    TestCase,
    TextSimilarityEvaluator, # Still from here, though its definition is unconfirmed
)
from src.core.metric_registry import MetricRegistry


@MetricRegistry.register("creativity")
class CreativityEvaluator(BaseMetricEvaluator):
    """Evaluator for model creativity capabilities."""
    
    def __init__(self, env: LLMEnvironmentProtocol, similarity_service: TextSimilarityEvaluator):
        self.env = env
        self.similarity_service = similarity_service
        
    def _calculate_novelty(self, text: str, reference_texts: List[str]) -> float:
        """Calculate novelty by comparing to reference texts."""
        if not reference_texts:
            return 1.0  # Maximum novelty if no references
            
        # Calculate similarity to each reference
        similarities = []
        for reference in reference_texts:
            similarity = self.similarity_service.calculate_similarity(text, reference)
            similarities.append(similarity)
            
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        return novelty
        
    def _calculate_diversity(self, responses: List[str]) -> Dict[str, float]:
        """Calculate diversity among multiple responses."""
        if len(responses) < 2:
            return {
                "pairwise_diversity": 0.0,
                "average_similarity": 0.0,
                "similarity_stdev": 0.0
            }
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self.similarity_service.calculate_similarity(responses[i], responses[j])
                similarities.append(similarity)
                
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        similarity_stdev = statistics.stdev(similarities) if len(similarities) > 1 else 0
        
        # Diversity is inverse of average similarity
        diversity = 1.0 - avg_similarity
        
        return {
            "pairwise_diversity": diversity,
            "average_similarity": avg_similarity,
            "similarity_stdev": similarity_stdev
        }
        
    def _calculate_surprise(self, text: str) -> float:
        """Calculate surprise factor based on uncommon patterns and language use."""
        # This is a simplified implementation - in a real system this would use
        # more sophisticated NLP techniques
        
        # Check for unusual word combinations
        unusual_patterns = [
            r"(?:\w+ly\s+\w+ly)",  # Double adverbs
            r"(?:[A-Z][a-z]+ed\s+[A-Z][a-z]+ing)",  # Past participle + present participle
            r"(?:[A-Z][a-z]+\s+of\s+[A-Z][a-z]+ness)",  # Noun of abstract noun
            r"(?:[A-Z][a-z]+ish\s+[A-Z][a-z]+esque)",  # Multiple adjective suffixes
            r"(?:un[A-Z][a-z]+\s+re[A-Z][a-z]+)"  # Multiple prefixed words
        ]
        
        unusual_count = 0
        for pattern in unusual_patterns:
            matches = re.findall(pattern, text)
            unusual_count += len(matches)
            
        # Check for sentence structure variation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return 0.0
            
        sentence_lengths = [len(s) for s in sentences]
        length_variation = statistics.stdev(sentence_lengths) / max(1, statistics.mean(sentence_lengths)) if len(sentence_lengths) > 1 else 0
        
        # Look for unusual punctuation patterns
        unusual_punctuation = len(re.findall(r"[;:—–…]", text)) / max(1, len(text))
        
        # Combine factors
        surprise_score = (
            min(1.0, unusual_count / 5) * 0.4 +  # Unusual patterns (cap at 5)
            min(1.0, length_variation) * 0.4 +  # Sentence length variation
            min(1.0, unusual_punctuation * 20) * 0.2  # Unusual punctuation (scaled)
        )
        
        return surprise_score
        
    def _calculate_relevance(self, response: str, prompt: str) -> float:
        """Calculate relevance of creative response to the original prompt."""
        relevance = self.similarity_service.calculate_similarity(response, prompt)
        return relevance
    
    async def evaluate(self, model_id: str, test_cases: List[TestCase]) -> MetricResult:
        """Evaluate model creativity capabilities."""
        creativity_scores = []
        details = []
        
        # First pass - collect all responses for diversity calculation
        all_responses = []
        for case in test_cases:
            prompt = case['input']
            response = await self.env.get_model_response(model_id, prompt)
            all_responses.append(response)
            
        # Calculate diversity across all responses
        diversity_results = self._calculate_diversity(all_responses)
        
        # Second pass - detailed creativity assessment
        for i, case in enumerate(test_cases):
            prompt = case['input']
            response = all_responses[i]
            expected_output = case.get('expected_output', '')
            
            # Get variations of the prompt for different responses
            variation_prompts = [
                f"{prompt} Please be creative and original in your response.",
                f"{prompt} Provide an unexpected or surprising answer."
            ]
            
            variation_responses = []
            for var_prompt in variation_prompts:
                var_response = await self.env.get_model_response(model_id, var_prompt)
                variation_responses.append(var_response)
            
            # All responses for this test case
            case_responses = [response] + variation_responses
            
            # Calculate novelty
            reference_texts = [expected_output] if expected_output else []
            novelty = self._calculate_novelty(response, reference_texts)
            
            # Calculate case-specific diversity
            case_diversity = self._calculate_diversity(case_responses)["pairwise_diversity"]
            
            # Calculate surprise factor
            surprise = self._calculate_surprise(response)
            
            # Calculate relevance
            relevance = self._calculate_relevance(response, prompt)
            
            # Calculate overall creativity score
            # Higher weights for novelty and diversity, balanced by relevance
            creativity = (
                novelty * 0.3 +
                case_diversity * 0.2 +
                surprise * 0.2 +
                diversity_results["pairwise_diversity"] * 0.1 +
                relevance * 0.2  # Relevance ensures creative responses are still on-topic
            )
            
            creativity_scores.append(creativity)
            
            details.append({
                'prompt': prompt,
                'response': response,
                'variation_responses': variation_responses,
                'novelty': novelty,
                'case_diversity': case_diversity,
                'surprise': surprise,
                'relevance': relevance,
                'creativity_score': creativity
            })
        
        overall_creativity = sum(creativity_scores) / len(creativity_scores) if creativity_scores else 0
        
        return {
            'value': overall_creativity,
            'details': {
                'case_details': details,
                'overall_creativity': overall_creativity,
                'cross_response_diversity': diversity_results
            }
        }
"""
