"""
Evaluation Metrics for LLM Outputs
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

class EvaluationMetrics:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        nltk.download('punkt')
    
    async def calculate_metrics(self, 
                              generated_text: str,
                              reference_text: str = None,
                              metrics: List[str] = None) -> Dict[str, float]:
        """Calculate various evaluation metrics."""
        if metrics is None:
            metrics = ['rouge', 'bleu', 'bert_score', 'coherence']
        
        results = {}
        
        for metric in metrics:
            if metric == 'rouge':
                results['rouge'] = self._calculate_rouge(generated_text, reference_text)
            elif metric == 'bleu':
                results['bleu'] = self._calculate_bleu(generated_text, reference_text)
            elif metric == 'bert_score':
                results['bert_score'] = await self._calculate_bert_score(
                    generated_text, reference_text
                )
            elif metric == 'coherence':
                results['coherence'] = await self._calculate_coherence(generated_text)
        
        return results
    
    def _calculate_rouge(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        return {key: value.fmeasure for key, value in scores.items()}
    
    def _calculate_bleu(self, generated_text: str, reference_text: str) -> float:
        """Calculate BLEU score."""
        reference = nltk.word_tokenize(reference_text)
        candidate = nltk.word_tokenize(generated_text)
        return sentence_bleu([reference], candidate)
    
    async def _calculate_bert_score(self, generated_text: str, reference_text: str) -> float:
        """Calculate BERTScore."""
        P, R, F1 = score([generated_text], [reference_text], lang='en', verbose=False)
        return F1.mean().item()
    
    async def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence using embeddings."""
        # Get embeddings for each sentence
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
        
        embeddings = await self._get_embeddings(sentences)
        
        # Calculate coherence as average cosine similarity between adjacent sentences
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores)