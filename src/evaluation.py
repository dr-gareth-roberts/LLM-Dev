"""
Evaluation Metrics for LLM Outputs

-------------------------------------------
DEPRECATION NOTICE
-------------------------------------------
This module is deprecated and will be removed in a future version.
Please use the new unified evaluation system in `src.evaluation_system` instead.

For migration assistance, see:
1. The migration_helper.py tool in src/tools/
2. Documentation in docs/evaluation_framework.md

Example migration:
```python
# Old usage:
from src.evaluation import EvaluationMetrics
evaluator = EvaluationMetrics(env)
results = await evaluator.calculate_metrics(...)

# New usage:
from src.evaluation_system import ModelEvaluator
evaluator = ModelEvaluator(env)
# For evaluating texts directly:
result = await evaluator.calculate_text_metrics(
    generated_text=generated_text,
    reference_text=reference_text,
    metrics=['rouge_score', 'bleu_score', 'bert_score']
)
```
-------------------------------------------
"""
from typing import List, Dict, Any
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import rouge_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

# Import the new system for delegation
from src.evaluation_system import ModelEvaluator
from src.tools.migration_helper import MigrationHelper

# Show deprecation warning
warnings.warn(
    "The evaluation module is deprecated. "
    "Please use src.evaluation_system instead.",
    DeprecationWarning,
    stacklevel=2
)

class EvaluationMetrics:
    """
    DEPRECATED: This class is maintained for backward compatibility.
    Please use src.evaluation_system.ModelEvaluator instead.
    """
    
    def __init__(self, env: 'LLMDevEnvironment'):
        warnings.warn(
            "This EvaluationMetrics class is deprecated. Use src.evaluation_system.ModelEvaluator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.env = env
        nltk.download('punkt')
        
        # Initialize the new evaluator for delegation
        self._new_evaluator = ModelEvaluator(env)
        self._migration_helper = MigrationHelper(env)
    
    async def calculate_metrics(self, 
                              generated_text: str,
                              reference_text: str = None,
                              metrics: List[str] = None) -> Dict[str, float]:
        """
        DEPRECATED: Calculate various evaluation metrics.
        This delegates to the new evaluation system while maintaining the old interface.
        """
        if metrics is None:
            metrics = ['rouge', 'bleu', 'bert_score', 'coherence']
        
        # Map old metric names to new ones
        mapped_metrics = []
        for metric in metrics:
            mapped_metric = self._migration_helper.metric_mapping.get(metric, metric)
            mapped_metrics.append(mapped_metric)
        
        # Create a simple test case with the generated and reference text
        test_case = {
            'input': 'placeholder_prompt',  # Not used for text evaluation
            'generated_output': generated_text,
            'expected_output': reference_text
        }
        
        # Delegate to new evaluator's text metric calculation
        results = await self._new_evaluator.calculate_text_metrics(
            generated_text=generated_text,
            reference_text=reference_text,
            metrics=mapped_metrics
        )
        
        # Convert results back to old format if needed
        old_format_results = {}
        for new_metric, score in results.items():
            # Find the original metric name (reverse mapping)
            original_metrics = [
                old for old, new in self._migration_helper.metric_mapping.items() 
                if new == new_metric
            ]
            old_metric = original_metrics[0] if original_metrics else new_metric
            
            if isinstance(score, dict):
                # Handle nested scores like ROUGE
                old_format_results[old_metric] = score
            else:
                old_format_results[old_metric] = score
        
        return old_format_results
    
    def _calculate_rouge(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        DEPRECATED: Calculate ROUGE scores.
        This is maintained for backward compatibility but delegates to the new system.
        """
        warnings.warn(
            "The _calculate_rouge method is deprecated. Use ModelEvaluator.calculate_text_metrics instead.",
            DeprecationWarning,
            stacklevel=2
        )
        scorer = rouge_score.rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        return {key: value.fmeasure for key, value in scores.items()}
    
    def _calculate_bleu(self, generated_text: str, reference_text: str) -> float:
        """
        DEPRECATED: Calculate BLEU score.
        This is maintained for backward compatibility but should use the new system.
        """
        warnings.warn(
            "The _calculate_bleu method is deprecated. Use ModelEvaluator.calculate_text_metrics instead.",
            DeprecationWarning,
            stacklevel=2
        )
        reference = nltk.word_tokenize(reference_text)
        candidate = nltk.word_tokenize(generated_text)
        return sentence_bleu([reference], candidate)
    
    async def _calculate_bert_score(self, generated_text: str, reference_text: str) -> float:
        """
        DEPRECATED: Calculate BERTScore.
        This is maintained for backward compatibility but should use the new system.
        """
        warnings.warn(
            "The _calculate_bert_score method is deprecated. Use ModelEvaluator.calculate_text_metrics instead.",
            DeprecationWarning,
            stacklevel=2
        )
        P, R, F1 = score([generated_text], [reference_text], lang='en', verbose=False)
        return F1.mean().item()
    
    async def _calculate_coherence(self, text: str) -> float:
        """
        DEPRECATED: Calculate text coherence using embeddings.
        This is maintained for backward compatibility but should use the new system.
        """
        warnings.warn(
            "The _calculate_coherence method is deprecated. Use ModelEvaluator.calculate_text_metrics instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
