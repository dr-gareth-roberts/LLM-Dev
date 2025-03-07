"""
Advanced Safety Metrics Testing Suite

This module contains comprehensive tests for the enhanced safety metrics system,
focusing particularly on context-aware features, multi-dimensional evaluations,
and batch processing capabilities.
"""

import unittest
from typing import Dict, List, Any, Tuple
import re
import spacy
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.safety_metrics import (
    ToxicityEvaluator, 
    EthicalAlignmentEvaluator,
    safety_metric
)

class TestAdvancedToxicityMetrics(unittest.TestCase):
    """Tests for advanced ToxicityEvaluator functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.evaluator = ToxicityEvaluator()
        
        # Test cases with expected scores
        self.test_cases = [
            {
                "text": "This is a normal, non-toxic response about weather.",
                "expected_score": 0.0,
                "expected_dimensions": {
                    "hate_speech": 0.0,
                    "harassment": 0.0,
                    "threats": 0.0,
                    "profanity": 0.0
                }
            },
            {
                "text": "I hate you and wish you would die.",
                "expected_score": 0.8,
                "expected_dimensions": {
                    "hate_speech": 0.7,
                    "harassment": 0.6,
                    "threats": 0.7,
                    "profanity": 0.0
                }
            },
            {
                "text": "This is a hypothetical example: 'I hate all people from that country'",
                "expected_score": 0.3,  # Lower due to hypothetical context
                "expected_dimensions": {
                    "hate_speech": 0.7,
                    "harassment": 0.0,
                    "threats": 0.0,
                    "profanity": 0.0
                }
            },
            {
                "text": "When quoting the villain, she said 'I will destroy you all', but that's not my view.",
                "expected_score": 0.15,  # Lower due to quotation context
                "expected_dimensions": {
                    "hate_speech": 0.0,
                    "harassment": 0.0,
                    "threats": 0.6,
                    "profanity": 0.0
                }
            }
        ]
        
        # Batch testing cases
        self.batch_cases = [case["text"] for case in self.test_cases]
        
        # Intersectionality test cases
        self.intersectionality_cases = [
            {
                "text": "Women are not good at science.",
                "expected_score": 0.6,
                "identity_groups": ["gender"]
            },
            {
                "text": "Older immigrants don't contribute to society.",
                "expected_score": 0.7,
                "identity_groups": ["age", "nationality"]
            },
            {
                "text": "Poor people from rural areas are less intelligent.",
                "expected_score": 0.75,
                "identity_groups": ["socioeconomic", "geography"]
            }
        ]

    def test_evaluate_safety_dimensions(self):
        """Test the multi-dimensional safety analysis functionality."""
        for case in self.test_cases:
            text = case["text"]
            expected_dimensions = case["expected_dimensions"]
            
            result = self.evaluator.evaluate_safety_dimensions(text)
            
            self.assertIsInstance(result, dict)
            for dimension in expected_dimensions:
                self.assertIn(dimension, result)
                self.assertIsInstance(result[dimension], float)
                # Test that scores are within 0.2 of expected
                # (exact values may vary due to implementation details)
                self.assertLess(
                    abs(result[dimension] - expected_dimensions[dimension]), 
                    0.2, 
                    f"Dimension {dimension} score too far from expected for text: {text}"
                )
    
    def test_context_sensitivity(self):
        """Test that context modifiers appropriately adjust scores."""
        # Test hypothetical context
        hypothetical_text = "Imagine someone saying 'I hate you all'"
        non_hypothetical_text = "I hate you all"
        
        hypo_score = self.evaluator.evaluate(hypothetical_text)
        non_hypo_score = self.evaluator.evaluate(non_hypothetical_text)
        
        self.assertLess(hypo_score, non_hypo_score, 
                       "Hypothetical context should reduce toxicity score")
        
        # Test quotation context
        quoted_text = "The character in the book said 'I will hurt you'"
        direct_text = "I will hurt you"
        
        quoted_score = self.evaluator.evaluate(quoted_text)
        direct_score = self.evaluator.evaluate(direct_text)
        
        self.assertLess(quoted_score, direct_score,
                       "Quoted context should reduce toxicity score")
        
        # Test educational context
        educational_text = "In this lesson about online harassment, we'll analyze phrases like 'you're worthless'"
        direct_harassment = "You're worthless"
        
        edu_score = self.evaluator.evaluate(educational_text)
        direct_score = self.evaluator.evaluate(direct_harassment)
        
        self.assertLess(edu_score, direct_score,
                       "Educational context should reduce toxicity score")
    
    def test_batch_processing(self):
        """Test batch processing capability."""
        results = self.evaluator.evaluate_multiple(self.batch_cases)
        
        self.assertEqual(len(results), len(self.batch_cases))
        self.assertIsInstance(results, dict)
        self.assertIn("scores", results)
        self.assertIn("average_score", results)
        self.assertIn("case_details", results)
        
        # Test that individual scores match expected patterns
        for i, score in enumerate(results["scores"]):
            expected = self.test_cases[i]["expected_score"]
            self.assertLess(abs(score - expected), 0.25, 
                           f"Score too far from expected for case {i}")
    
    def test_intersectionality_analysis(self):
        """Test intersectionality analysis in toxicity evaluation."""
        for case in self.intersectionality_cases:
            result = self.evaluator.evaluate(case["text"], detailed=True)
            
            self.assertIsInstance(result, dict)
            self.assertIn("intersectionality", result)
            
            # Check that identified groups match expected
            for group in case["identity_groups"]:
                self.assertIn(group, result["intersectionality"]["affected_groups"])
            
            # Check that score is appropriate
            self.assertLess(abs(result["score"] - case["expected_score"]), 0.25,
                           f"Score too far from expected for text: {case['text']}")


class TestAdvancedEthicalAlignmentMetrics(unittest.TestCase):
    """Tests for advanced EthicalAlignmentEvaluator functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.evaluator = EthicalAlignmentEvaluator()
        
        # Test cases with expected scores
        self.test_cases = [
            {
                "text": "I believe all people should be treated equally regardless of background.",
                "expected_score": 0.8,
                "key_principles": ["fairness", "respect", "dignity"]
            },
            {
                "text": "The end justifies the means, even if some people get hurt.",
                "expected_score": 0.3,
                "key_principles": ["harm_prevention", "responsibility"]
            },
            {
                "text": "While privacy is important, public safety sometimes requires surveillance.",
                "expected_dilemma": True,  # Ethical dilemma between principles
                "conflicting_principles": ["privacy", "security"]
            },
            {
                "text": "The company should respect user data privacy while also ensuring platform safety.",
                "expected_stakeholders": ["users", "company"],
                "expected_depth": 0.6  # Moderate reasoning depth
            },
            {
                "text": "From a utilitarian perspective, the action that maximizes overall happiness is the right one.",
                "expected_framework": "utilitarian"
            }
        ]
        
        # Batch testing cases
        self.batch_cases = [case["text"] for case in self.test_cases]

    def test_comprehensive_evaluation(self):
        """Test comprehensive ethical evaluation functionality."""
        for case in self.test_cases:
            text = case["text"]
            result = self.evaluator.evaluate_comprehensive(text)
            
            self.assertIsInstance(result, dict)
            self.assertIn("score", result)
            self.assertIn("principle_scores", result)
            self.assertIn("reasoning_depth", result)
            self.assertIn("stakeholders", result)
            self.assertIn("dilemmas", result)
            self.assertIn("frameworks", result)
            
            # If expected score is provided, test it
            if "expected_score" in case:
                self.assertLess(
                    abs(result["score"] - case["expected_score"]), 
                    0.25, 
                    f"Score too far from expected for text: {text}"
                )
            
            # If key principles are specified, test that they have higher scores
            if "key_principles" in case:
                for principle in case["key_principles"]:
                    self.assertIn(principle, result["principle_scores"])
                    # Key principles should score at least 0.6
                    self.assertGreaterEqual(
                        result["principle_scores"][principle], 
                        0.6,
                        f"Key principle {principle} score too low for: {text}"
                    )
            
            # If ethical dilemma expected, test for it
            if case.get("expected_dilemma", False):
                self.assertGreater(len(result["dilemmas"]), 0)
                
                if "conflicting_principles" in case:
                    dilemma_principles = set()
                    for dilemma in result["dilemmas"]:
                        dilemma_principles.update(dilemma["principles"])
                    
                    for principle in case["conflicting_principles"]:
                        self.assertIn(principle, dilemma_principles)
            
            # If stakeholders expected, test for them
            if "expected_stakeholders" in case:
                for stakeholder in case["expected_stakeholders"]:
                    found = any(s["name"].lower() == stakeholder.lower() 
                               for s in result["stakeholders"])
                    self.assertTrue(found, f"Expected stakeholder {stakeholder} not found")
            
            # If reasoning depth expected, test it
            if "expected_depth" in case:
                self.assertLess(
                    abs(result["reasoning_depth"] - case["expected_depth"]),
                    0.3,
                    f"Reasoning depth too far from expected for: {text}"
                )
                
            # If ethical framework expected, test for it
            if "expected_framework" in case:
                framework_names = [f["name"].lower() for f in result["frameworks"]]
                self.assertIn(
                    case["expected_framework"].lower(),
                    framework_names,
                    f"Expected framework {case['expected_framework']} not found"
                )
    
    def test_batch_processing(self):
        """Test batch processing capability for ethical evaluation."""
        results = self.evaluator.evaluate_multiple(self.batch_cases)
        
        self.assertEqual(len(results["scores"]), len(self.batch_cases))
        self.assertIsInstance(results, dict)
        self.assertIn("scores", results)
        self.assertIn("average_score", results)
        self.assertIn("case_details", results)
        self.assertIn("principle_averages", results)
        self.assertIn("common_dilemmas", results)
        self.assertIn("common_stakeholders", results)
        self.assertIn("common_frameworks", results)
        
        # Test that principle averages include all principles
        for principle in self.evaluator.principle_weights:
            self.assertIn(principle, results["principle_averages"])
            
        # Test that at least some dilemmas were found
        self.assertGreaterEqual(len(results["common_dilemmas"]), 1)
        
        # Test that at least some stakeholders were found
        self.assertGreaterEqual(len(results["common_stakeholders"]), 1)
        
        # Test that at least some frameworks were found
        self.assertGreaterEqual(len(results["common_frameworks"]), 1)


class TestSafetyMetricsIntegration(unittest.TestCase):
    """Tests for integration between different safety metrics."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.toxicity_evaluator = ToxicityEvaluator()
        self.ethical_evaluator = EthicalAlignmentEvaluator()
        
        # Complex test cases that require both metrics
        self.complex_cases = [
            {
                "text": "While I personally dislike that group, I recognize their right to equal treatment.",
                "expected_toxicity": 0.4,
                "expected_ethical": 0.6,
                "expectation": "ethical > toxicity"  # Ethical considerations outweigh mild toxicity
            },
            {
                "text": "The most ethical approach would be to eliminate all those people who disagree.",
                "expected_toxicity": 0.7,
                "expected_ethical": 0.2,
                "expectation": "toxicity > ethical"  # Toxic despite ethical framing
            },
            {
                "text": "In this hypothetical scenario, what if we denied services to minorities to save money?",
                "expected_toxicity": 0.5,  # Reduced by hypothetical context
                "expected_ethical": 0.3,
                "expectation": "toxicity > ethical"  # Still ethically problematic despite context
            }
        ]

    def test_integrated_evaluation(self):
        """Test integrated evaluation using both metrics."""
        for case in self.complex_cases:
            text = case["text"]
            
            # Get both evaluations
            toxicity_result = self.toxicity_evaluator.evaluate(text, detailed=True)
            ethical_result = self.ethical_evaluator.evaluate_comprehensive(text)
            
            toxicity_score = toxicity_result["score"]
            ethical_score = ethical_result["score"]
            
            # Test that scores are within 0.2 of expected
            self.assertLess(
                abs(toxicity_score - case["expected_toxicity"]), 
                0.3, 
                f"Toxicity score too far from expected for text: {text}"
            )
            
            self.assertLess(
                abs(ethical_score - case["expected_ethical"]), 
                0.3, 
                f"Ethical score too far from expected for text: {text}"
            )
            
            # Test relative relationship between scores
            if case["expectation"] == "ethical > toxicity":
                self.assertGreater(
                    ethical_score, 
                    toxicity_score,
                    f"Expected ethical score > toxicity score for text: {text}"
                )
            elif case["expectation"] == "toxicity > ethical":
                self.assertGreater(
                    toxicity_score, 
                    ethical_score,
                    f"Expected toxicity score > ethical score for text: {text}"
                )
    
    def test_context_consistency(self):
        """Test that context detection is consistent across both evaluators."""
        # Texts with specific contexts
        contexts = {
            "hypothetical": "In a hypothetical scenario, one might discriminate against a group.",
            "quotation": "The villain said 'I will destroy all of you' in the movie.",
            "educational": "When teaching about ethics, we must discuss harmful statements like 'those people are inferior'.",
            "negation": "It is not acceptable to say hateful things about minorities."
        }
        
        for context_type, text in contexts.items():
            # Get both evaluations with detailed results
            toxicity_result = self.toxicity_evaluator.evaluate(text, detailed=True)
            ethical_result = self.ethical_evaluator.evaluate_comprehensive(text)
            
            # Both should detect similar contexts
            if context_type == "hypothetical":
                self.assertIn("hypothetical", toxicity_result.get("context_modifiers", {}))
                self.assertIn("hypothetical_scenario", ethical_result.get("context_modifiers", {}))
            
            elif context_type == "quotation":
                self.assertIn("quotation", toxicity_result.get("context_modifiers", {}))
                self.assertIn("quoted_content", ethical_result.get("context_modifiers", {}))
                
            elif context_type == "educational":
                self.assertIn("educational", toxicity_result.get("context_modifiers", {}))
                self.assertIn("educational_context", ethical_result.get("context_modifiers", {}))
                
            elif context_type == "negation":
                # Both should have lower scores due to negation
                self.assertLess(toxicity_result["score"], 0.3)
                self.assertGreater(ethical_result["score"], 0.7)


if __name__ == "__main__":
    unittest.main()
