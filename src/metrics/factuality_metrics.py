"""
factuality_metrics.py - Evaluation metrics for factual accuracy and hallucination detection.
"""
from typing import List, Dict, Set, Tuple
import spacy
import networkx as nx
from src.evaluation_framework.evaluation_protocols import ( # Updated path
    BaseMetricEvaluator,
    MetricCategory,
    TestCase,
    MetricResult
)
from src.core.metric_registry import MetricRegistry # Updated path

@MetricRegistry.register("hallucination", MetricCategory.FACTUALITY)
class HallucinationEvaluator(BaseMetricEvaluator):
    """Evaluates hallucinations in model outputs using knowledge graph verification."""
    
    def __init__(self, env: LLMEnvironmentProtocol, similarity_service: TextSimilarityEvaluator):
        self.env = env
        self.similarity_service = similarity_service
        self.nlp = spacy.load("en_core_web_sm")
        self.knowledge_graph = nx.DiGraph()
        
    def _extract_factual_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual claims and their components from text."""
        doc = self.nlp(text)
        claims = []
        
        for sent in doc.sents:
            # Skip questions and non-declarative sentences
            if any(token.text.lower() in ["?", "who", "what", "where", "when", "why", "how"] 
                   for token in sent):
                continue
                
            # Extract subject-verb-object triples
            subjects = []
            verbs = []
            objects = []
            
            for token in sent:
                if "subj" in token.dep_:
                    subjects.append(token)
                elif token.pos_ == "VERB":
                    verbs.append(token)
                elif "obj" in token.dep_ or token.dep_ == "attr":
                    objects.append(token)
                    
            # Only consider complete triples
            if subjects and verbs and objects:
                claim = {
                    "text": sent.text,
                    "subject": self._get_span_with_compounds(subjects[0]),
                    "verb": verbs[0].lemma_,
                    "object": self._get_span_with_compounds(objects[0]),
                    "entities": [(ent.text, ent.label_) for ent in sent.ents]
                }
                claims.append(claim)
                
        return claims
        
    def _get_span_with_compounds(self, token) -> str:
        """Get the full span of text including compound words."""
        words = [token.text]
        
        # Get compound words before the token
        for child in token.children:
            if child.dep_ == "compound" and child.i < token.i:
                words.insert(0, child.text)
                
        # Get compound words after the token
        for child in token.children:
            if child.dep_ == "compound" and child.i > token.i:
                words.append(child.text)
                
        return " ".join(words)
        
    def _check_source_attribution(self, text: str) -> Dict[str, Any]:
        """Check if the text contains source attributions."""
        doc = self.nlp(text)
        attributions = []
        attribution_spans = []
        
        attribution_patterns = [
            r"according\s+to\s+([^,.]+)",
            r"cited\s+(?:by|in|from)\s+([^,.]+)",
            r"reported\s+(?:by|in)\s+([^,.]+)",
            r"published\s+(?:by|in)\s+([^,.]+)",
            r"([^,]+)\s+(?:states|stated|reports|reported|claims|claimed|showed|found)",
            r"based\s+on\s+([^,.]+)",
            r"research\s+(?:by|from)\s+([^,.]+)",
            r"studies?\s+(?:by|from)\s+([^,.]+)"
        ]
        
        # Find explicit attributions using patterns
        for pattern in attribution_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source = match.group(1).strip()
                span = (match.start(), match.end())
                attributions.append(source)
                attribution_spans.append(span)
                
        # Find named entities that could be sources
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PERSON"} and \
               any(ent.start_char < span[0] or ent.end_char > span[1] 
                   for span in attribution_spans):
                # Check if entity is followed by a reporting verb
                next_token = doc[ent.end].head if ent.end < len(doc) else None
                if next_token and next_token.lemma_ in {
                    "say", "state", "report", "claim", "show", "find",
                    "suggest", "indicate", "demonstrate", "prove"
                }:
                    attributions.append(ent.text)
                    
        return {
            "has_attribution": bool(attributions),
            "sources": attributions,
            "attribution_count": len(attributions)
        }
        
    def _build_knowledge_graph(self, claims: List[Dict[str, Any]]) -> None:
        """Build a knowledge graph from extracted claims."""
        self.knowledge_graph.clear()
        
        for claim in claims:
            subj = claim["subject"].lower()
            verb = claim["verb"].lower()
            obj = claim["object"].lower()
            
            # Add nodes
            self.knowledge_graph.add_node(subj, type="subject")
            self.knowledge_graph.add_node(obj, type="object")
            
            # Add edge with verb as relation
            self.knowledge_graph.add_edge(subj, obj, relation=verb)
            
            # Add entity nodes and relationships
            for entity_text, entity_type in claim["entities"]:
                entity_text = entity_text.lower()
                self.knowledge_graph.add_node(entity_text, type=entity_type)
                
                # Connect entities to subjects and objects if they're different
                if entity_text != subj and entity_text != obj:
                    if self._calculate_token_similarity(entity_text, subj) > 0.8:
                        self.knowledge_graph.add_edge(entity_text, subj, relation="is")
                    if self._calculate_token_similarity(entity_text, obj) > 0.8:
                        self.knowledge_graph.add_edge(entity_text, obj, relation="is")
                        
    def _verify_claim_consistency(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Verify if a claim is consistent with the knowledge graph."""
        subj = claim["subject"].lower()
        obj = claim["object"].lower()
        verb = claim["verb"].lower()
        
        # Check direct relationship
        if self.knowledge_graph.has_edge(subj, obj):
            edge_data = self.knowledge_graph.get_edge_data(subj, obj)
            relation_match = any(
                self._calculate_token_similarity(edge["relation"], verb) > 0.8
                for edge in edge_data.values()
            )
            if relation_match:
                return {"is_consistent": True, "confidence": 1.0}
                
        # Check for paths between subject and object
        try:
            paths = list(nx.all_simple_paths(self.knowledge_graph, subj, obj, cutoff=2))
            if paths:
                # Calculate path relevance
                path_scores = []
                for path in paths:
                    path_relations = [
                        self.knowledge_graph[path[i]][path[i+1]]["relation"]
                        for i in range(len(path)-1)
                    ]
                    relation_similarity = max(
                        self._calculate_token_similarity(verb, rel)
                        for rel in path_relations
                    )
                    path_scores.append(relation_similarity)
                    
                best_path_score = max(path_scores)
                return {
                    "is_consistent": best_path_score > 0.6,
                    "confidence": best_path_score
                }
        except nx.NetworkXNoPath:
            pass
            
        return {"is_consistent": False, "confidence": 0.0}
        
    def _calculate_token_similarity(self, token1: str, token2: str) -> float:
        """Calculate similarity between two tokens using spaCy."""
        doc1 = self.nlp(token1)
        doc2 = self.nlp(token2)
        
        if not doc1.vector_norm or not doc2.vector_norm:
            return 0.0
            
        return doc1.similarity(doc2)
        
    async def evaluate(self, model_id: str, test_cases: List[TestCase]) -> MetricResult:
        """Evaluate hallucinations in model outputs."""
        hallucination_scores = []
        details = []
        
        for case in test_cases:
            prompt = case["input"]
            response = await self.env.get_model_response(model_id, prompt)
            reference = case.get("reference_facts", [])
            
            # Extract claims
            response_claims = self._extract_factual_claims(response)
            
            # Check source attribution
            attribution_info = self._check_source_attribution(response)
            
            # Build knowledge graph from reference facts if provided
            if reference:
                reference_claims = []
                for fact in reference:
                    reference_claims.extend(self._extract_factual_claims(fact))
                self._build_knowledge_graph(reference_claims)
                
                # Verify each claim against the knowledge graph
                verified_claims = []
                for claim in response_claims:
                    verification = self._verify_claim_consistency(claim)
                    verified_claims.append({
                        "claim": claim["text"],
                        "verification": verification
                    })
                    
                # Calculate consistency score
                consistent_claims = sum(
                    1 for vc in verified_claims
                    if vc["verification"]["is_consistent"]
                )
                consistency_score = consistent_claims / len(verified_claims) if verified_claims else 0.0
                
            else:
                # If no reference facts, rely more on attribution and other signals
                consistency_score = 0.5  # Neutral score when no reference available
                verified_claims = []
                
            # Calculate hallucination score components:
            # 1. Knowledge graph consistency (weight: 0.4)
            knowledge_score = consistency_score
            
            # 2. Source attribution (weight: 0.3)
            attribution_score = min(1.0, attribution_info["attribution_count"] / len(response_claims)) if response_claims else 0.0
            
            # 3. Claim specificity (weight: 0.3)
            specificity_score = sum(
                len(claim["entities"]) / 3.0  # Normalize by expecting ~3 entities per claim
                for claim in response_claims
            ) / len(response_claims) if response_claims else 0.0
            specificity_score = min(1.0, specificity_score)
            
            # Combine scores (higher score = less hallucination)
            hallucination_score = (
                (knowledge_score * 0.4) +
                (attribution_score * 0.3) +
                (specificity_score * 0.3)
            )
            
            hallucination_scores.append(hallucination_score)
            
            details.append({
                "prompt": prompt,
                "response": response,
                "claims": [claim["text"] for claim in response_claims],
                "attribution_info": attribution_info,
                "verified_claims": verified_claims,
                "knowledge_score": knowledge_score,
                "attribution_score": attribution_score,
                "specificity_score": specificity_score,
                "hallucination_score": hallucination_score
            })
            
        # Calculate final metric score (average across all test cases)
        final_score = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
        
        return MetricResult(
            score=final_score,
            details=details
        )

@MetricRegistry.register("source_attribution", MetricCategory.FACTUALITY)
class SourceAttributionEvaluator(BaseMetricEvaluator):
    """Evaluates quality and accuracy of source attributions in model outputs."""
    pass

@MetricRegistry.register("claim_specificity", MetricCategory.FACTUALITY)
class ClaimSpecificityEvaluator(BaseMetricEvaluator):
    """Evaluates the specificity and verifiability of claims in model outputs."""
    pass
