"""
Tests for factuality evaluation metrics.

These tests verify the functionality of the hallucination detection metrics.
"""
import pytest
import asyncio
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.factuality_metrics import HallucinationEvaluator


class TestHallucinationMetric:
    """Tests for the HallucinationMetric class."""
    
    @pytest.mark.asyncio
    async def test_factual_response_with_knowledge_graph(self, mock_env):
        """Test detection of factual content using knowledge graph verification."""
        metric = HallucinationEvaluator(mock_env)
        
        test_case = {
            "input": "What is the relationship between Oxford and the UK?",
            "model_response": "Oxford University is located in Oxford, which is a city in the United Kingdom.",
            "reference_facts": [
                "Oxford is a city in England",
                "England is part of the United Kingdom",
                "The University of Oxford is located in Oxford"
            ],
            "category": "factual"
        }
        
        result = await metric.evaluate([test_case])
        
        assert result.score > 0.8  # Should score high for factual content
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["knowledge_score"] > 0.8  # High knowledge graph consistency
        assert len(detail["claims"]) >= 2  # Should extract multiple claims
        
    @pytest.mark.asyncio
    async def test_knowledge_graph_contradiction(self, mock_env):
        """Test detection of contradictions using knowledge graph."""
        metric = HallucinationEvaluator(mock_env)
        
        test_case = {
            "input": "Tell me about Einstein's education",
            "model_response": "Einstein studied at Oxford University in England before developing relativity.",
            "reference_facts": [
                "Einstein studied at the Swiss Federal Polytechnic School",
                "Einstein never studied at Oxford University",
                "Einstein developed the theory of relativity while working as a patent clerk"
            ],
            "category": "factual"
        }
        
        result = await metric.evaluate([test_case])
        
        assert result.score < 0.4  # Should score low due to contradiction
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["knowledge_score"] < 0.4  # Low knowledge graph consistency
        
    @pytest.mark.asyncio
    async def test_source_attribution_quality(self, mock_env):
        """Test evaluation of source attribution quality."""
        metric = HallucinationEvaluator(mock_env)
        
        # Test with strong attribution
        well_attributed_case = {
            "input": "What are the effects of climate change?",
            "model_response": "According to the IPCC's 2023 report, global temperatures continue to rise. Research from NASA shows increased frequency of extreme weather events. Studies by climate scientists at Oxford University demonstrate rising sea levels.",
            "category": "factual"
        }
        
        # Test with weak attribution
        poorly_attributed_case = {
            "input": "What are the effects of climate change?",
            "model_response": "Global temperatures are definitely rising very rapidly. Extreme weather is becoming much more common everywhere. Sea levels are rising at unprecedented rates.",
            "category": "factual"
        }
        
        well_attributed_result = await metric.evaluate([well_attributed_case])
        poorly_attributed_result = await metric.evaluate([poorly_attributed_case])
        
        assert well_attributed_result.score > poorly_attributed_result.score
        assert well_attributed_result.details[0]["attribution_score"] > 0.7
        assert poorly_attributed_result.details[0]["attribution_score"] < 0.3
        
    @pytest.mark.asyncio
    async def test_claim_specificity(self, mock_env):
        """Test evaluation of claim specificity using named entities."""
        metric = HallucinationEvaluator(mock_env)
        
        # Test with specific claims
        specific_case = {
            "input": "What happened in the 2022 FIFA World Cup final?",
            "model_response": "Argentina defeated France in the 2022 FIFA World Cup final at Lusail Stadium in Qatar. Lionel Messi led Argentina to victory with a 4-2 penalty shootout win after a 3-3 draw.",
            "category": "factual"
        }
        
        # Test with vague claims
        vague_case = {
            "input": "What happened in the World Cup final?",
            "model_response": "The team played well and won the match after extra time. The players celebrated with their fans after scoring more goals.",
            "category": "factual"
        }
        
        specific_result = await metric.evaluate([specific_case])
        vague_result = await metric.evaluate([vague_case])
        
        assert specific_result.score > vague_result.score
        assert specific_result.details[0]["specificity_score"] > 0.7
        assert vague_result.details[0]["specificity_score"] < 0.3
        
    @pytest.mark.asyncio
    async def test_multi_hop_knowledge_verification(self, mock_env):
        """Test verification of claims requiring multi-hop reasoning in knowledge graph."""
        metric = HallucinationEvaluator(mock_env)
        
        test_case = {
            "input": "What is the relationship between Shakespeare and the English monarchy?",
            "model_response": "Shakespeare wrote plays during Queen Elizabeth I's reign in England.",
            "reference_facts": [
                "William Shakespeare was a playwright in England",
                "Shakespeare wrote plays from 1589 to 1613",
                "Elizabeth I was Queen of England from 1558 to 1603",
                "Elizabeth I was part of the Tudor dynasty"
            ],
            "category": "factual"
        }
        
        result = await metric.evaluate([test_case])
        
        assert result.score > 0.7  # Should verify through multiple knowledge graph hops
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["knowledge_score"] > 0.7  # Should establish connection through multiple facts
        
    @pytest.mark.asyncio
    async def test_entity_relationship_verification(self, mock_env):
        """Test verification of relationships between named entities."""
        metric = HallucinationEvaluator(mock_env)
        
        test_case = {
            "input": "What is the relationship between Python and Guido van Rossum?",
            "model_response": "Guido van Rossum created Python while working at CWI in the Netherlands.",
            "reference_facts": [
                "Guido van Rossum is the creator of Python",
                "Python was created at CWI",
                "CWI is located in the Netherlands",
                "Python was first released in 1991"
            ],
            "category": "factual"
        }
        
        result = await metric.evaluate([test_case])
        
        assert result.score > 0.8  # Should verify entity relationships correctly
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail["knowledge_score"] > 0.8  # High knowledge graph consistency for entity relationships
