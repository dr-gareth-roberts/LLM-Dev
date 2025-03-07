# LLM-Dev Evaluation Framework

## Overview

The LLM-Dev Evaluation Framework provides a comprehensive, plugin-based architecture for evaluating Large Language Models (LLMs) across multiple dimensions. This document outlines the framework architecture, available metrics, extension points, and usage guidelines.

## Architecture

The evaluation framework is built on a modular, extensible architecture with the following core components:

### Core Components

1. **Evaluation Protocols** (`evaluation_protocols.py`)
   - Defines protocol interfaces and type definitions used throughout the system
   - Includes `MetricResult`, `TestCase`, and other TypedDict definitions
   - Provides Protocol classes for standardised interfaces
   - Includes abstract base classes for metrics and evaluators

2. **Evaluation System** (`evaluation_system.py`)
   - Implements a unified evaluation system with plugin architecture
   - Includes `MetricRegistry` for dynamic metric discovery and registration
   - Provides `TextSimilarityService` for embeddings and similarity calculations
   - Features `ModelEvaluator` as the main entry point for evaluation

3. **Metric Plugins**
   - Implemented in separate modules for organisation and extensibility
   - Each metric implements standardised interfaces for consistent behaviour
   - Metrics are automatically discovered and registered with the registry

## Available Metrics

The framework includes the following metric categories:

### Basic Metrics
- **Accuracy**: Evaluates the precision of model responses against expected outputs
- **Consistency**: Measures the model's ability to maintain consistent answers across similar queries

### Content Safety Metrics
- **Toxicity Detection**: Identifies harmful, offensive, or inappropriate content in model outputs
- **Bias Evaluation**: Assesses model outputs for various forms of bias, including gender, race, and cultural biases

### Factuality Metrics
- **Hallucination Detection**: Identifies instances where the model generates factually incorrect information
- **Knowledge Verification**: Validates model outputs against known facts and information

### Performance Metrics
- **Robustness Testing**: Evaluates model performance under adversarial conditions, including typos and malformed inputs
- **Efficiency Measurement**: Measures token usage, response time, and information density

### Cognitive Metrics
- **Reasoning Assessment**: Evaluates the model's logical reasoning capabilities
- **Instruction Following**: Measures how well the model follows multi-part instructions
- **Creativity Evaluation**: Assesses originality, novelty, and relevance of creative outputs

## Usage

### Command Line Interface

The evaluation framework is integrated into the main CLI with the `evaluate` command:

```bash
python main.py evaluate <model_id> [--test-file FILE] [--metrics METRIC1 METRIC2 ...]
```

**Parameters:**
- `model_id`: The identifier of the model to evaluate
- `--test-file` or `-t`: Path to a YAML file containing test cases
- `--metrics` or `-m`: Specific metrics to evaluate (can be specified multiple times)

### Programmatic Usage

```python
from src.evaluation_system import ModelEvaluator
from llm_environment import LLMDevEnvironment

# Initialize environment and evaluator
env = LLMDevEnvironment()
evaluator = ModelEvaluator(env)

# Define test cases
test_cases = [
    {
        "input": "What is the capital of France?",
        "expected_output": "Paris",
        "category": "factual",
        "difficulty": "easy"
    }
]

# Run evaluation
result = await evaluator.evaluate_model(
    model_id="gpt-4",
    test_cases=test_cases,
    metrics=["accuracy", "hallucination_detection"]
)

# Access results
print(f"Overall score: {result.overall_score}")
for metric, score in result.metrics.items():
    print(f"{metric}: {score}")
```

## Extending the Framework

### Creating Custom Metrics

1. Create a new Python module in the `src/metrics` directory
2. Define a class that inherits from `BaseMetric` or implements the `MetricProtocol`
3. Implement the required methods, especially `compute`
4. Add your metric to the `__init__.py` file in the metrics directory for automatic registration

Example:

```python
from src.evaluation_protocols import MetricResult, TestCaseWithResponse
from src.metrics.base import BaseMetric

class MyCustomMetric(BaseMetric):
    """My custom evaluation metric."""
    
    name = "my_custom_metric"
    description = "Evaluates model performance using my custom approach"
    version = "1.0.0"
    
    async def compute(self, test_case: TestCaseWithResponse) -> MetricResult:
        # Implement your metric logic here
        score = self._calculate_score(test_case)
        
        return {
            "score": score,
            "details": {
                "custom_field": "Additional information"
            }
        }
    
    def _calculate_score(self, test_case):
        # Custom calculation logic
        return 0.85  # Example score
```

## Best Practices

1. **Test Case Design**
   - Include diverse test cases covering various scenarios
   - Specify appropriate difficulty levels and categories
   - Include edge cases and adversarial examples

2. **Metric Selection**
   - Choose metrics appropriate for your use case
   - Combine complementary metrics for a comprehensive evaluation
   - Consider the computational cost of resource-intensive metrics

3. **Results Interpretation**
   - Consider both overall scores and individual metric results
   - Analyse detailed metric outputs for deeper insights
   - Compare results across different models and test cases

## Technical Requirements

The evaluation framework requires the dependencies listed in `requirements-evaluation.txt`. Install with:

```bash
pip install -r requirements-evaluation.txt
```

For development and testing of custom metrics, use the development requirements:

```bash
pip install -r requirements-dev.txt
```
