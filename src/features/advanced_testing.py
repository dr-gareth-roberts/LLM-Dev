"""
Advanced Testing Module for LLM Development Environment
"""
import asyncio
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

class AdvancedTesting:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.logger = env.logger
        self.results_dir = env.env_config.output_dir / 'advanced_testing'
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_benchmark(self, 
                          prompts: List[str],
                          models: List[Dict[str, Any]],
                          metrics: List[str] = None):
        """
        Run comprehensive benchmark tests across different models and prompts.
        """
        if metrics is None:
            metrics = ['latency', 'token_usage', 'cost', 'consistency']
            
        results = []
        for prompt in prompts:
            for model_config in models:
                result = await self._test_single_config(prompt, model_config, metrics)
                results.append(result)
                
        return self._analyze_results(results)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _test_single_config(self, prompt: str, model_config: Dict[str, Any], metrics: List[str]):
        """
        Test a single prompt-model configuration with retries.
        """
        start_time = datetime.now()
        try:
            # Run the model
            response = await self._run_model(prompt, model_config)
            
            # Calculate metrics
            metrics_result = {
                'latency': (datetime.now() - start_time).total_seconds(),
                'token_usage': self._calculate_token_usage(prompt, response),
                'cost': self._calculate_cost(response, model_config),
                'consistency': await self._measure_consistency(prompt, model_config),
                'response': response
            }
            
            return {
                'prompt': prompt,
                'model': model_config['name'],
                'timestamp': start_time.isoformat(),
                'metrics': metrics_result,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in test: {str(e)}")
            return {
                'prompt': prompt,
                'model': model_config['name'],
                'timestamp': start_time.isoformat(),
                'error': str(e),
                'success': False
            }
    
    async def run_parameter_sweep(self, 
                                prompt: str,
                                parameter_ranges: Dict[str, List[float]],
                                model: str = "gpt-4"):
        """
        Perform parameter sweep to find optimal settings.
        """
        results = []
        total_combinations = np.prod([len(values) for values in parameter_ranges.values()])
        
        for i in range(int(total_combinations)):
            # Generate parameter combination
            params = {}
            for param_name, param_values in parameter_ranges.items():
                idx = (i // np.prod([len(parameter_ranges[p]) for p in list(parameter_ranges.keys())[:list(parameter_ranges.keys()).index(param_name)])) % len(param_values)
                params[param_name] = param_values[idx]
            
            # Test configuration
            result = await self._test_parameters(prompt, model, params)
            results.append(result)
        
        return self._analyze_parameter_sweep(results)
    
    async def evaluate_consistency(self, prompt: str, num_runs: int = 5):
        """
        Evaluate model consistency across multiple runs.
        """
        results = []
        for _ in range(num_runs):
            response = await self._run_model(prompt, {'name': 'gpt-4'})
            results.append(response)
        
        return self._analyze_consistency(results)
    
    async def compare_embeddings(self, texts: List[str], models: List[str] = None):
        """
        Compare embeddings from different models.
        """
        if models is None:
            models = ['openai', 'sentence-transformers']
        
        embeddings = {}
        for model in models:
            embeddings[model] = await self._get_embeddings(texts, model)
        
        return self._analyze_embeddings(embeddings)
    
    def generate_report(self, results: Dict[str, Any], report_type: str):
        """
        Generate comprehensive PDF report of test results.
        """
        report_path = self.results_dir / f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create visualizations
        figs = self._create_visualizations(results)
        
        # Generate PDF report
        self._generate_pdf_report(results, figs, report_path)
        
        return report_path
    
    def _create_visualizations(self, results: Dict[str, Any]) -> List[plt.Figure]:
        """
        Create various visualizations of test results.
        """
        figs = []
        
        # Latency comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(results['latencies']), ax=ax)
        ax.set_title('Latency Comparison Across Models')
        figs.append(fig)
        
        # Cost comparison
        if 'costs' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=pd.DataFrame(results['costs']), ax=ax)
            ax.set_title('Cost Comparison Across Models')
            figs.append(fig)
        
        # Add more visualizations as needed
        
        return figs