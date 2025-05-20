"""
test_metric_registry.py - Tests for the metric registry system.

This module contains comprehensive tests for the metric registry functionality,
ensuring proper registration, retrieval, and categorisation of metrics.
"""
import pytest
from typing import List
from src.core.metric_registry import MetricRegistry # Updated path
from src.evaluation_framework.evaluation_protocols import ( # Updated path
    BaseMetricEvaluator,
    MetricCategory,
    LLMEnvironmentProtocol,
    TestCase,
    MetricResult
)


class MockEnvironment(LLMEnvironmentProtocol):
    """Mock LLM environment for testing."""
    async def get_response(self, prompt: str) -> str:
        return "mock response"


class MockMetric(BaseMetricEvaluator):
    """Mock metric evaluator for testing."""
    async def evaluate(self, model_id: str, test_cases: List[TestCase]) -> MetricResult:
        return MetricResult(
            metric_name="mock_metric",
            score=1.0,
            details=[]
        )


@pytest.fixture
def env():
    """Fixture providing mock environment."""
    return MockEnvironment()


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    MetricRegistry.clear()
    yield
    MetricRegistry.clear()


def test_register_metric():
    """Test basic metric registration."""
    @MetricRegistry.register("mock", MetricCategory.FACTUALITY)
    class TestMetric(MockMetric):
        pass

    assert "mock" in MetricRegistry._metrics
    assert MetricRegistry._categories["mock"] == MetricCategory.FACTUALITY


def test_register_duplicate_metric():
    """Test registering duplicate metric raises error."""
    @MetricRegistry.register("mock", MetricCategory.FACTUALITY)
    class TestMetric1(MockMetric):
        pass

    with pytest.raises(ValueError):
        @MetricRegistry.register("mock", MetricCategory.SAFETY)
        class TestMetric2(MockMetric):
            pass


def test_register_duplicate_metric_with_override():
    """Test registering duplicate metric with override."""
    @MetricRegistry.register("mock", MetricCategory.FACTUALITY)
    class TestMetric1(MockMetric):
        pass

    @MetricRegistry.register("mock", MetricCategory.SAFETY, override=True)
    class TestMetric2(MockMetric):
        pass

    assert MetricRegistry._categories["mock"] == MetricCategory.SAFETY


def test_get_metric(env):
    """Test retrieving registered metric."""
    @MetricRegistry.register("mock", MetricCategory.FACTUALITY)
    class TestMetric(MockMetric):
        pass

    metric = MetricRegistry.get_metric("mock", env)
    assert isinstance(metric, TestMetric)


def test_get_nonexistent_metric(env):
    """Test retrieving non-existent metric raises error."""
    with pytest.raises(KeyError):
        MetricRegistry.get_metric("nonexistent", env)


def test_list_metrics():
    """Test listing all registered metrics."""
    @MetricRegistry.register("mock1", MetricCategory.FACTUALITY)
    class TestMetric1(MockMetric):
        pass

    @MetricRegistry.register("mock2", MetricCategory.SAFETY)
    class TestMetric2(MockMetric):
        pass

    metrics = MetricRegistry.list_metrics()
    assert set(metrics) == {"mock1", "mock2"}


def test_list_metrics_by_category():
    """Test listing metrics filtered by category."""
    @MetricRegistry.register("mock1", MetricCategory.FACTUALITY)
    class TestMetric1(MockMetric):
        pass

    @MetricRegistry.register("mock2", MetricCategory.SAFETY)
    class TestMetric2(MockMetric):
        pass

    @MetricRegistry.register("mock3", MetricCategory.FACTUALITY)
    class TestMetric3(MockMetric):
        pass

    metrics = MetricRegistry.list_metrics(MetricCategory.FACTUALITY)
    assert set(metrics) == {"mock1", "mock3"}


def test_get_category():
    """Test getting metric category."""
    @MetricRegistry.register("mock", MetricCategory.FACTUALITY)
    class TestMetric(MockMetric):
        pass

    category = MetricRegistry.get_category("mock")
    assert category == MetricCategory.FACTUALITY


def test_get_nonexistent_category():
    """Test getting category of non-existent metric raises error."""
    with pytest.raises(KeyError):
        MetricRegistry.get_category("nonexistent")


def test_clear_registry():
    """Test clearing registry."""
    @MetricRegistry.register("mock", MetricCategory.FACTUALITY)
    class TestMetric(MockMetric):
        pass

    MetricRegistry.clear()
    assert not MetricRegistry._metrics
    assert not MetricRegistry._categories


def test_register_batch():
    """Test registering multiple metrics at once."""
    class TestMetric1(MockMetric):
        pass

    class TestMetric2(MockMetric):
        pass

    metrics = {
        "mock1": TestMetric1,
        "mock2": TestMetric2
    }
    categories = {
        "mock1": MetricCategory.FACTUALITY,
        "mock2": MetricCategory.SAFETY
    }

    MetricRegistry.register_batch(metrics, categories)
    assert "mock1" in MetricRegistry._metrics
    assert "mock2" in MetricRegistry._metrics
    assert MetricRegistry._categories["mock1"] == MetricCategory.FACTUALITY
    assert MetricRegistry._categories["mock2"] == MetricCategory.SAFETY


def test_register_batch_missing_category():
    """Test registering batch with missing category raises error."""
    class TestMetric(MockMetric):
        pass

    metrics = {"mock": TestMetric}
    categories = {}

    with pytest.raises(ValueError):
        MetricRegistry.register_batch(metrics, categories)
