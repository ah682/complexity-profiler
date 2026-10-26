"""Collected regression checks for the Phase 1 exception, logging and metrics APIs."""

import logging
import math
from types import SimpleNamespace

import pytest

from complexity_profiler.analysis import metrics as metrics_module
from complexity_profiler.analysis.metrics import (
    AggregateMetricsCollector,
    DefaultMetricsCollector,
    PerformanceMetrics,
    ThreadSafeMetricsCollector,
    measure_time,
)
from complexity_profiler.utils.exceptions import (
    AlgorithmError,
    BigOComplexityError,
    ConfigurationError,
    CurveFittingError,
    DataGenerationError,
    ExportError,
    ProfilingError,
)
from complexity_profiler.utils.logging_config import get_logger, set_module_level, setup_logging


@pytest.mark.parametrize(
    "error_type",
    [
        AlgorithmError,
        DataGenerationError,
        ProfilingError,
        CurveFittingError,
        ConfigurationError,
        ExportError,
    ],
)
def test_exception_imports_and_hierarchy(error_type: type[BigOComplexityError]) -> None:
    with pytest.raises(BigOComplexityError) as caught:
        raise error_type("import and hierarchy check")
    assert caught.value.message == "import and hierarchy check"


def test_base_exception_details_and_algorithm_context() -> None:
    assert str(BigOComplexityError("Test error", details={"key": "value"})) == (
        "Test error (key=value)"
    )
    error = AlgorithmError("Algorithm failed", algorithm_name="quicksort")
    assert error.algorithm_name == "quicksort"


def test_logging_configuration() -> None:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    module_logger = get_logger("phase1_verification")
    module_level = module_logger.level
    try:
        assert setup_logging(level="INFO", enable_rich=False) is root
        assert any(handler.level == logging.INFO for handler in root.handlers)
        assert get_logger("phase1_verification") is module_logger
        set_module_level("phase1_verification", "DEBUG")
        assert module_logger.level == logging.DEBUG
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers = handlers
        root.setLevel(level)
        module_logger.setLevel(module_level)


def test_performance_metrics_total_operations() -> None:
    metrics = PerformanceMetrics(comparisons=10, swaps=5)
    assert metrics.comparisons == 10
    assert metrics.total_operations() == 15


@pytest.mark.parametrize("collector_type", [DefaultMetricsCollector, ThreadSafeMetricsCollector])
def test_collectors_record_operations_and_timing(
    collector_type: type[DefaultMetricsCollector] | type[ThreadSafeMetricsCollector],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([20.0, 20.125])
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    collector = collector_type()
    collector.record_comparison()
    collector.record_swap()
    with collector.timing():
        collector.record_access()
    result = collector.get_metrics()
    assert (result.comparisons, result.swaps, result.accesses) == (1, 1, 1)
    assert result.execution_time == 0.125


@pytest.mark.parametrize("collector_type", [DefaultMetricsCollector, ThreadSafeMetricsCollector])
def test_collector_protocol_operations_and_reset(
    collector_type: type[DefaultMetricsCollector] | type[ThreadSafeMetricsCollector],
) -> None:
    collector = collector_type()
    for method in (
        "record_comparison",
        "record_swap",
        "record_access",
        "record_recursive_call",
        "get_metrics",
        "reset",
    ):
        assert callable(getattr(collector, method))
    collector.record_comparison()
    collector.record_swap()
    collector.record_access()
    collector.record_recursive_call()
    assert collector.get_metrics().total_operations() == 4
    collector.reset()
    assert collector.get_metrics().total_operations() == 0


def test_aggregate_metrics_collector() -> None:
    aggregate = AggregateMetricsCollector()
    for comparisons in (1, 2, 3):
        collector = DefaultMetricsCollector()
        for _ in range(comparisons):
            collector.record_comparison()
        aggregate.add_metrics(collector.get_metrics())
    summary = aggregate.get_summary()
    assert summary["runs"] == 3
    assert summary["mean_comparisons"] == 2.0


@pytest.mark.parametrize("end", [41.0, 41.125])
def test_measure_time_records_clock_difference_on_context_exit(
    monkeypatch: pytest.MonkeyPatch, end: float
) -> None:
    ticks = iter([41.0, end])
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    with measure_time() as elapsed:
        assert elapsed == [0.0]
    assert elapsed == [end - 41.0]


def test_measure_time_records_elapsed_time_when_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([80.0, 80.25])
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    with pytest.raises(ValueError, match="body failed"):
        with measure_time() as elapsed:
            raise ValueError("body failed")
    assert elapsed == [0.25]


def test_measure_time_real_clock_smoke() -> None:
    with measure_time() as elapsed:
        result = sum(range(100))
    assert result == 4950
    assert len(elapsed) == 1
    assert math.isfinite(elapsed[0])
    assert elapsed[0] >= 0
