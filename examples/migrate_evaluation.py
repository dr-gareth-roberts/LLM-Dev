#!/usr/bin/env python3
"""
migrate_evaluation.py - Example script demonstrating migration from legacy evaluation systems

This script demonstrates how to:
1. Convert legacy test cases and evaluation results to the new format
2. Update code that uses the old evaluation classes to use the new unified system
3. Combine metrics from different evaluation files into a single unified approach
"""

import asyncio
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import yaml
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import legacy and new evaluation modules
from src.evaluation_framework.advanced_evaluation import LLMDevEnvironment # Updated path
from src.evaluation_framework.evaluation import EvaluationMetrics  # Legacy module, updated path
from src.evaluation_framework.advanced_evaluation import ModelEvaluator as LegacyModelEvaluator  # Legacy module, updated path
from src.evaluation_framework.evaluation_system import ModelEvaluator  # New unified system, updated path
from src.tools.migration_helper import MigrationHelper # Path is correct


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


async def example_migrate_text_evaluation():
    """Example showing migration from legacy EvaluationMetrics to the new system."""
    logger.info("Migrating from legacy EvaluationMetrics to new unified system...")
    
    # Initialize environment
    env = LLMDevEnvironment()
    
    # Example text to evaluate
    generated_text = "The quick brown fox jumps over the lazy dog."
    reference_text = "A fast brown fox leaps over a lazy dog."
    
    # =====================================
    # Legacy approach - EvaluationMetrics
    # =====================================
    logger.info("Using legacy EvaluationMetrics...")
    legacy_metrics = EvaluationMetrics(env)
    legacy_results = await legacy_metrics.calculate_metrics(
        generated_text=generated_text,
        reference_text=reference_text,
        metrics=['rouge', 'bleu', 'bert_score']
    )
    logger.info(f"Legacy results: {legacy_results}")
    
    # =====================================
    # New unified approach - ModelEvaluator
    # =====================================
    logger.info("Using new unified ModelEvaluator...")
    unified_evaluator = ModelEvaluator(env)
    new_results = await unified_evaluator.calculate_text_metrics(
        generated_text=generated_text,
        reference_text=reference_text,
        metrics=['rouge_score', 'bleu_score', 'bert_score']
    )
    logger.info(f"New unified results: {new_results}")
    
    logger.info("Text evaluation migration complete!")


async def example_migrate_model_evaluation():
    """Example showing migration from legacy ModelEvaluator to the new system."""
    logger.info("Migrating from legacy ModelEvaluator to new unified system...")
    
    # Initialize environment
    env = LLMDevEnvironment()
    
    # Example test cases
    test_cases = [
        {
            "input": "What is the capital of France?",
            "expected_output": "The capital of France is Paris.",
            "category": "factual",
            "difficulty": "easy"
        },
        {
            "input": "Explain quantum computing in simple terms.",
            "expected_output": "Quantum computing uses quantum mechanics to perform calculations faster than classical computers.",
            "category": "explanation",
            "difficulty": "medium"
        }
    ]
    
    model_id = "gpt-4"  # Example model ID
    
    # =====================================
    # Legacy approach - ModelEvaluator
    # =====================================
    logger.info("Using legacy ModelEvaluator...")
    legacy_evaluator = LegacyModelEvaluator(env)
    legacy_results = await legacy_evaluator.evaluate_model(
        model_id=model_id,
        test_cases=test_cases,
        metrics=['accuracy', 'consistency']
    )
    logger.info(f"Legacy accuracy: {legacy_results.metrics.get('accuracy')}")
    
    # =====================================
    # New unified approach - ModelEvaluator
    # =====================================
    logger.info("Using new unified ModelEvaluator...")
    unified_evaluator = ModelEvaluator(env)
    new_results = await unified_evaluator.evaluate_model(
        model_id=model_id,
        test_cases=test_cases,
        metrics=['accuracy', 'consistency']
    )
    logger.info(f"New unified accuracy: {new_results.metrics.get('accuracy')}")
    
    logger.info("Model evaluation migration complete!")


async def example_migrate_test_cases():
    """Example showing migration of test case files."""
    logger.info("Demonstrating test case file migration...")
    
    # Initialize environment and migration helper
    env = LLMDevEnvironment()
    migration_helper = MigrationHelper(env)
    
    # Create a temporary legacy test case file
    legacy_test_cases = {
        "test_cases": [
            {
                "prompt": "What is the capital of France?",
                "expected_response": "The capital of France is Paris.",
                "category": "factual"
            },
            {
                "prompt": "Write a poem about spring.",
                "expected_response": "",  # Open-ended
                "category": "creative"
            }
        ]
    }
    
    # Save legacy test cases to a file
    legacy_file_path = Path("./legacy_test_cases.yaml")
    with open(legacy_file_path, "w") as f:
        yaml.dump(legacy_test_cases, f)
    
    logger.info(f"Created legacy test case file at {legacy_file_path}")
    
    # Migrate the test cases
    new_file_path = migration_helper.convert_test_cases(legacy_file_path)
    
    if new_file_path:
        logger.info(f"Successfully migrated test cases to {new_file_path}")
        
        # Show the content of the migrated file
        with open(new_file_path, "r") as f:
            migrated_test_cases = yaml.safe_load(f)
        
        logger.info("Migrated test case format:")
        for case in migrated_test_cases["test_cases"]:
            logger.info(f"  Input: {case.get('input')}")
            logger.info(f"  Expected output: {case.get('expected_output')}")
            logger.info(f"  Category: {case.get('category')}")
            logger.info("  ---")
    
    # Clean up temporary files
    if legacy_file_path.exists():
        legacy_file_path.unlink()
    if new_file_path and Path(new_file_path).exists():
        Path(new_file_path).unlink()
    
    logger.info("Test case migration example complete!")


async def example_full_migration():
    """Run a complete migration example demonstrating all aspects."""
    logger.info("Starting full migration example...")
    
    # Text evaluation migration
    await example_migrate_text_evaluation()
    print("\n" + "=" * 80 + "\n")
    
    # Model evaluation migration
    await example_migrate_model_evaluation()
    print("\n" + "=" * 80 + "\n")
    
    # Test case file migration
    await example_migrate_test_cases()
    print("\n" + "=" * 80 + "\n")
    
    logger.info("Full migration example completed successfully!")
    logger.info("""
Migration Summary:
1. Legacy EvaluationMetrics -> New ModelEvaluator.calculate_text_metrics()
2. Legacy ModelEvaluator -> New ModelEvaluator with compatible interface
3. Legacy test case files -> New format test case files

The new unified evaluation system provides all functionality from both
legacy systems while adding plugin-based metrics, better organisation,
and improved extensibility.
""")


if __name__ == "__main__":
    asyncio.run(example_full_migration())
