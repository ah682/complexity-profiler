"""The installed public API must import without referencing nonexistent names."""


def test_public_metrics_aliases():
    from complexity_profiler import Metrics, MetricsCollector
    from complexity_profiler.algorithms.base import MetricsCollector as Protocol
    from complexity_profiler.analysis.metrics import PerformanceMetrics

    assert Metrics is PerformanceMetrics
    assert MetricsCollector is Protocol
