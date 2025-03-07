"""
migration_helper.py - Utilities to help migrate from legacy evaluation systems to the new unified system

This module provides tools to:
1. Convert legacy evaluation results to the new format
2. Migrate test cases to the new format
3. Map old metric names to new metric plugins
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import yaml

from ..evaluation_protocols import TestCase, EvaluationResult
from ..evaluation_system import ModelEvaluator

logger = logging.getLogger(__name__)


class MigrationHelper:
    """Helper class for migrating from legacy evaluation systems to the new unified system."""
    
    def __init__(self, env: 'LLMDevEnvironment'):
        """
        Initialize the migration helper.
        
        Args:
            env: The LLM development environment
        """
        self.env = env
        self.legacy_results_dir = env.env_config.output_dir / 'evaluations'
        self.new_results_dir = env.env_config.output_dir / 'unified_evaluations'
        self.new_results_dir.mkdir(exist_ok=True)
        
        # Mapping from old metric names to new metric plugins
        self.metric_mapping = {
            'accuracy': 'accuracy',
            'consistency': 'consistency',
            'toxicity': 'toxicity',
            'bias': 'bias',
            'hallucination': 'hallucination_detection',
            'robustness': 'robustness',
            'efficiency': 'efficiency',
            'reasoning': 'reasoning',
            'instruction_following': 'instruction_following',
            'creativity': 'creativity',
            'rouge': 'rouge_score',
            'bleu': 'bleu_score',
            'bert_score': 'bert_score',
            'coherence': 'coherence'
        }
    
    def convert_legacy_result(self, legacy_result_path: Union[str, Path]) -> Optional[EvaluationResult]:
        """
        Convert a legacy evaluation result to the new format.
        
        Args:
            legacy_result_path: Path to the legacy evaluation result JSON file
            
        Returns:
            Converted EvaluationResult object or None if conversion failed
        """
        path = Path(legacy_result_path)
        if not path.exists():
            logger.error(f"Legacy result file not found: {path}")
            return None
        
        try:
            with open(path, 'r') as f:
                legacy_data = json.load(f)
            
            # Create new evaluation result with mapped metrics
            new_metrics = {}
            for old_metric, old_value in legacy_data.get('metrics', {}).items():
                new_metric = self.metric_mapping.get(old_metric, old_metric)
                new_metrics[new_metric] = old_value
            
            # Convert detailed results
            new_details = {}
            for old_metric, old_details in legacy_data.get('detailed_results', {}).items():
                new_metric = self.metric_mapping.get(old_metric, old_metric)
                new_details[new_metric] = old_details
            
            # Create new result object
            new_result = {
                'model_id': legacy_data.get('model_id', 'unknown'),
                'timestamp': datetime.fromisoformat(legacy_data.get('timestamp')) 
                            if isinstance(legacy_data.get('timestamp'), str) 
                            else datetime.now(),
                'metrics': new_metrics,
                'detailed_results': new_details,
                'metadata': legacy_data.get('metadata', {}),
                'error_analysis': legacy_data.get('error_analysis', {})
            }
            
            # Save the converted result
            new_file_path = self.new_results_dir / f"{path.stem}_migrated.json"
            with open(new_file_path, 'w') as f:
                json.dump(new_result, f, indent=2, default=str)
            
            logger.info(f"Successfully converted {path} to {new_file_path}")
            return EvaluationResult(**new_result)
            
        except Exception as e:
            logger.error(f"Error converting legacy result {path}: {str(e)}")
            return None
    
    def migrate_all_results(self) -> Dict[str, int]:
        """
        Migrate all legacy evaluation results to the new format.
        
        Returns:
            Dictionary with counts of successful and failed migrations
        """
        results = {'successful': 0, 'failed': 0, 'skipped': 0}
        
        # Find all legacy result files
        for path in self.legacy_results_dir.glob('*.json'):
            # Skip already migrated files
            migrated_path = self.new_results_dir / f"{path.stem}_migrated.json"
            if migrated_path.exists():
                logger.info(f"Skipping already migrated result: {path}")
                results['skipped'] += 1
                continue
            
            # Convert the file
            if self.convert_legacy_result(path) is not None:
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def convert_test_cases(self, legacy_test_file: Union[str, Path]) -> Optional[Path]:
        """
        Convert legacy test cases to the new format.
        
        Args:
            legacy_test_file: Path to the legacy test cases file (YAML or JSON)
            
        Returns:
            Path to the converted test cases file or None if conversion failed
        """
        path = Path(legacy_test_file)
        if not path.exists():
            logger.error(f"Legacy test file not found: {path}")
            return None
        
        try:
            # Load legacy test cases
            legacy_test_cases = []
            if path.suffix.lower() in ('.yml', '.yaml'):
                with open(path, 'r') as f:
                    legacy_data = yaml.safe_load(f)
                    legacy_test_cases = legacy_data.get('test_cases', [])
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    legacy_data = json.load(f)
                    legacy_test_cases = legacy_data.get('test_cases', [])
            else:
                logger.error(f"Unsupported test file format: {path.suffix}")
                return None
            
            # Convert test cases to new format
            new_test_cases = []
            for case in legacy_test_cases:
                # Handle various legacy formats
                if 'input' in case:
                    # Already in new format
                    new_test_cases.append(case)
                elif 'prompt' in case:
                    # Old format with prompt
                    new_case = {
                        'input': case['prompt'],
                        'expected_output': case.get('expected_response', ''),
                        'category': case.get('category', 'general'),
                        'difficulty': case.get('difficulty', 'medium')
                    }
                    
                    # Copy any additional fields
                    for key, value in case.items():
                        if key not in ('prompt', 'expected_response'):
                            new_case.setdefault(key, value)
                    
                    new_test_cases.append(new_case)
            
            # Save converted test cases
            new_file_path = path.with_name(f"{path.stem}_migrated{path.suffix}")
            
            if path.suffix.lower() in ('.yml', '.yaml'):
                with open(new_file_path, 'w') as f:
                    yaml.dump({'test_cases': new_test_cases}, f, sort_keys=False)
            else:  # JSON
                with open(new_file_path, 'w') as f:
                    json.dump({'test_cases': new_test_cases}, f, indent=2)
            
            logger.info(f"Successfully converted {path} to {new_file_path}")
            return new_file_path
            
        except Exception as e:
            logger.error(f"Error converting test cases {path}: {str(e)}")
            return None
    
    def get_metric_mapping_guide(self) -> Dict[str, str]:
        """
        Get a guide for mapping old metric names to new metric plugins.
        
        Returns:
            Dictionary with old metric names as keys and new metric names as values
        """
        return self.metric_mapping


# Helper functions for direct use

def convert_legacy_result_file(env, file_path: str) -> Optional[str]:
    """
    Convenience function to convert a single legacy result file.
    
    Args:
        env: LLM development environment
        file_path: Path to the legacy result file
        
    Returns:
        Path to the converted file or None if conversion failed
    """
    helper = MigrationHelper(env)
    result = helper.convert_legacy_result(file_path)
    if result:
        migrated_path = helper.new_results_dir / f"{Path(file_path).stem}_migrated.json"
        return str(migrated_path)
    return None


def convert_legacy_test_file(env, file_path: str) -> Optional[str]:
    """
    Convenience function to convert a single legacy test file.
    
    Args:
        env: LLM development environment
        file_path: Path to the legacy test file
        
    Returns:
        Path to the converted file or None if conversion failed
    """
    helper = MigrationHelper(env)
    result = helper.convert_test_cases(file_path)
    return str(result) if result else None
