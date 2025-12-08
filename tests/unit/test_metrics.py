"""
Unit tests for metrics collection functionality.

This module tests the PerformanceMetrics dataclass, DefaultMetricsCollector,
and AggregateMetricsCollector classes to ensure accurate tracking of
algorithm operations and performance measurements.
"""

import pytest
import time
from typing import Any

from complexity_profiler.analysis.metrics import (
    PerformanceMetrics,
    DefaultMetricsCollector,
    AggregateMetricsCollector,
)


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""

    def test_initialization_defaults(self) -> None:
        """Test that PerformanceMetrics initializes with correct default values."""
        metrics = PerformanceMetrics()

        assert metrics.comparisons == 0
        assert metrics.swaps == 0
        assert metrics.accesses == 0
        assert metrics.recursive_calls == 0
        assert metrics.execution_time == 0.0
        assert metrics.memory_operations == 0
        assert metrics.memory_usage is None
        assert metrics.custom_metrics == {}

    def test_initialization_with_values(self) -> None:
        """Test that PerformanceMetrics initializes with provided values."""
        metrics = PerformanceMetrics(
            comparisons=100,
            swaps=50,
            accesses=200,
            recursive_calls=10,
            execution_time=1.5,
            memory_operations=25,
        )

        assert metrics.comparisons == 100
        assert metrics.swaps == 50
        assert metrics.accesses == 200
        assert metrics.recursive_calls == 10
        assert metrics.execution_time == 1.5
        assert metrics.memory_operations == 25

    def test_total_operations(self) -> None:
        """Test calculation of total operations."""
        metrics = PerformanceMetrics(
            comparisons=100,
            swaps=50,
            accesses=200,
            recursive_calls=10,
            memory_operations=25,
        )

        total = metrics.total_operations()

        assert total == 100 + 50 + 200 + 10 + 25
        assert total == 385

    def test_total_operations_zero(self) -> None:
        """Test total operations when all counts are zero."""
        metrics = PerformanceMetrics()

        assert metrics.total_operations() == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            comparisons=100,
            swaps=50,
            execution_time=1.5,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["comparisons"] == 100
        assert result["swaps"] == 50
        assert result["execution_time"] == 1.5
        assert result["accesses"] == 0

    def test_add_custom_metric(self) -> None:
        """Test adding custom metrics."""
        metrics = PerformanceMetrics()

        metrics.add_custom_metric("cache_hits", 42)
        metrics.add_custom_metric("cache_misses", 8)

        assert metrics.custom_metrics["cache_hits"] == 42
        assert metrics.custom_metrics["cache_misses"] == 8

    def test_add_custom_metric_overwrites(self) -> None:
        """Test that adding a custom metric with the same name overwrites."""
        metrics = PerformanceMetrics()

        metrics.add_custom_metric("count", 10)
        metrics.add_custom_metric("count", 20)

        assert metrics.custom_metrics["count"] == 20

    def test_merge_basic(self) -> None:
        """Test merging two PerformanceMetrics instances."""
        metrics1 = PerformanceMetrics(comparisons=100, swaps=50, execution_time=1.0)
        metrics2 = PerformanceMetrics(comparisons=200, swaps=75, execution_time=2.0)

        metrics1.merge(metrics2)

        assert metrics1.comparisons == 300
        assert metrics1.swaps == 125
        assert metrics1.execution_time == 3.0

    def test_merge_memory_usage(self) -> None:
        """Test that merge takes maximum memory usage."""
        metrics1 = PerformanceMetrics(memory_usage=1000)
        metrics2 = PerformanceMetrics(memory_usage=2000)

        metrics1.merge(metrics2)

        assert metrics1.memory_usage == 2000

    def test_merge_memory_usage_none(self) -> None:
        """Test merging when one has None memory usage."""
        metrics1 = PerformanceMetrics(memory_usage=None)
        metrics2 = PerformanceMetrics(memory_usage=1000)

        metrics1.merge(metrics2)

        assert metrics1.memory_usage == 1000

    def test_merge_custom_metrics_numeric(self) -> None:
        """Test merging custom numeric metrics."""
        metrics1 = PerformanceMetrics()
        metrics1.add_custom_metric("count", 10)

        metrics2 = PerformanceMetrics()
        metrics2.add_custom_metric("count", 20)

        metrics1.merge(metrics2)

        assert metrics1.custom_metrics["count"] == 30

    def test_merge_custom_metrics_non_numeric(self) -> None:
        """Test merging custom non-numeric metrics."""
        metrics1 = PerformanceMetrics()
        metrics1.add_custom_metric("status", "running")

        metrics2 = PerformanceMetrics()
        metrics2.add_custom_metric("status", "complete")

        metrics1.merge(metrics2)

        # Non-numeric values should not be summed
        assert metrics1.custom_metrics["status"] == "running"

    def test_str_representation(self) -> None:
        """Test string representation of metrics."""
        metrics = PerformanceMetrics(comparisons=100, swaps=50)

        result = str(metrics)

        assert "comparisons=100" in result
        assert "swaps=50" in result
        assert "PerformanceMetrics" in result

    def test_str_with_custom_metrics(self) -> None:
        """Test string representation includes custom metrics."""
        metrics = PerformanceMetrics()
        metrics.add_custom_metric("test_metric", 42)

        result = str(metrics)

        assert "custom=" in result
        assert "test_metric=42" in result


class TestDefaultMetricsCollector:
    """Test suite for DefaultMetricsCollector."""

    def test_initialization(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test that collector initializes with zero counts."""
        metrics = metrics_collector.get_metrics()

        assert metrics.comparisons == 0
        assert metrics.swaps == 0
        assert metrics.accesses == 0
        assert metrics.recursive_calls == 0
        assert metrics.execution_time == 0.0

    def test_record_comparison(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording comparison operations."""
        metrics_collector.record_comparison()
        metrics_collector.record_comparison()
        metrics_collector.record_comparison()

        metrics = metrics_collector.get_metrics()

        assert metrics.comparisons == 3

    def test_record_swap(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording swap operations."""
        metrics_collector.record_swap()
        metrics_collector.record_swap()

        metrics = metrics_collector.get_metrics()

        assert metrics.swaps == 2

    def test_record_access(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording access operations."""
        for _ in range(5):
            metrics_collector.record_access()

        metrics = metrics_collector.get_metrics()

        assert metrics.accesses == 5

    def test_record_recursive_call(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording recursive calls."""
        for _ in range(10):
            metrics_collector.record_recursive_call()

        metrics = metrics_collector.get_metrics()

        assert metrics.recursive_calls == 10

    def test_record_memory_operation(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording memory operations."""
        for _ in range(3):
            metrics_collector.record_memory_operation()

        metrics = metrics_collector.get_metrics()

        assert metrics.memory_operations == 3

    def test_record_custom_metric(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording custom metrics."""
        metrics_collector.record_custom_metric("test_value", 123)

        metrics = metrics_collector.get_metrics()

        assert metrics.custom_metrics["test_value"] == 123

    def test_set_memory_usage(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test setting memory usage."""
        metrics_collector.set_memory_usage(4096)

        metrics = metrics_collector.get_metrics()

        assert metrics.memory_usage == 4096

    def test_timing_context_manager(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test timing using context manager."""
        with metrics_collector.timing():
            time.sleep(0.01)  # Sleep for 10ms

        metrics = metrics_collector.get_metrics()

        assert metrics.execution_time >= 0.01
        assert metrics.execution_time < 0.1  # Should be less than 100ms

    def test_timing_context_manager_with_exception(
        self, metrics_collector: DefaultMetricsCollector
    ) -> None:
        """Test that timing context manager handles exceptions properly."""
        try:
            with metrics_collector.timing():
                time.sleep(0.01)
                raise ValueError("Test exception")
        except ValueError:
            pass

        metrics = metrics_collector.get_metrics()

        # Time should still be recorded even when exception occurs
        assert metrics.execution_time >= 0.01

    def test_start_stop_timing(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test manual start and stop timing."""
        metrics_collector.start_timing()
        time.sleep(0.01)
        elapsed = metrics_collector.stop_timing()

        assert elapsed >= 0.01
        metrics = metrics_collector.get_metrics()
        assert metrics.execution_time >= 0.01

    def test_stop_timing_without_start_raises_error(
        self, metrics_collector: DefaultMetricsCollector
    ) -> None:
        """Test that stopping timing without starting raises an error."""
        with pytest.raises(RuntimeError, match="start_timing"):
            metrics_collector.stop_timing()

    def test_get_execution_time_while_running(
        self, metrics_collector: DefaultMetricsCollector
    ) -> None:
        """Test getting execution time while timer is running."""
        metrics_collector.start_timing()
        time.sleep(0.01)

        elapsed = metrics_collector.get_execution_time()

        assert elapsed >= 0.01
        metrics_collector.stop_timing()

    def test_get_execution_time_after_stop(
        self, metrics_collector: DefaultMetricsCollector
    ) -> None:
        """Test getting execution time after timing has stopped."""
        metrics_collector.start_timing()
        time.sleep(0.01)
        metrics_collector.stop_timing()

        elapsed = metrics_collector.get_execution_time()

        assert elapsed >= 0.01

    def test_reset(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test resetting all metrics to zero."""
        # Record some operations
        metrics_collector.record_comparison()
        metrics_collector.record_swap()
        metrics_collector.record_access()
        with metrics_collector.timing():
            time.sleep(0.01)

        # Reset
        metrics_collector.reset()

        # All metrics should be zero
        metrics = metrics_collector.get_metrics()
        assert metrics.comparisons == 0
        assert metrics.swaps == 0
        assert metrics.accesses == 0
        assert metrics.execution_time == 0.0

    def test_multiple_operations(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test recording multiple different operations."""
        metrics_collector.record_comparison()
        metrics_collector.record_comparison()
        metrics_collector.record_swap()
        metrics_collector.record_access()
        metrics_collector.record_access()
        metrics_collector.record_access()
        metrics_collector.record_recursive_call()

        metrics = metrics_collector.get_metrics()

        assert metrics.comparisons == 2
        assert metrics.swaps == 1
        assert metrics.accesses == 3
        assert metrics.recursive_calls == 1
        assert metrics.total_operations() == 7

    def test_str_representation(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test string representation of collector."""
        metrics_collector.record_comparison()
        metrics_collector.record_swap()

        result = str(metrics_collector)

        assert "comparisons=1" in result
        assert "swaps=1" in result

    def test_repr_representation(self, metrics_collector: DefaultMetricsCollector) -> None:
        """Test repr representation of collector."""
        result = repr(metrics_collector)

        assert "DefaultMetricsCollector" in result
        assert "metrics=" in result


class TestAggregateMetricsCollector:
    """Test suite for AggregateMetricsCollector."""

    def test_initialization(self) -> None:
        """Test that aggregate collector initializes empty."""
        collector = AggregateMetricsCollector()

        assert collector.get_all_metrics() == []

    def test_add_single_metrics(self) -> None:
        """Test adding a single metrics instance."""
        collector = AggregateMetricsCollector()
        metrics = PerformanceMetrics(comparisons=100, swaps=50)

        collector.add_metrics(metrics)

        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) == 1
        assert all_metrics[0].comparisons == 100

    def test_add_multiple_metrics(self) -> None:
        """Test adding multiple metrics instances."""
        collector = AggregateMetricsCollector()

        for i in range(5):
            metrics = PerformanceMetrics(comparisons=i * 10, swaps=i * 5)
            collector.add_metrics(metrics)

        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) == 5

    def test_get_summary_empty(self) -> None:
        """Test getting summary with no metrics."""
        collector = AggregateMetricsCollector()

        summary = collector.get_summary()

        assert summary["runs"] == 0
        assert summary["mean_comparisons"] == 0.0
        assert summary["mean_swaps"] == 0.0

    def test_get_summary_single_run(self) -> None:
        """Test getting summary with single run."""
        collector = AggregateMetricsCollector()
        metrics = PerformanceMetrics(
            comparisons=100, swaps=50, accesses=200, execution_time=1.5
        )
        collector.add_metrics(metrics)

        summary = collector.get_summary()

        assert summary["runs"] == 1
        assert summary["mean_comparisons"] == 100.0
        assert summary["mean_swaps"] == 50.0
        assert summary["mean_accesses"] == 200.0
        assert summary["mean_execution_time"] == 1.5

    def test_get_summary_multiple_runs(self) -> None:
        """Test getting summary with multiple runs."""
        collector = AggregateMetricsCollector()

        # Add 3 runs with different values
        collector.add_metrics(PerformanceMetrics(comparisons=100, swaps=50))
        collector.add_metrics(PerformanceMetrics(comparisons=200, swaps=60))
        collector.add_metrics(PerformanceMetrics(comparisons=300, swaps=70))

        summary = collector.get_summary()

        assert summary["runs"] == 3
        assert summary["mean_comparisons"] == 200.0  # (100+200+300)/3
        assert summary["mean_swaps"] == 60.0  # (50+60+70)/3

    def test_get_summary_calculates_total_operations(self) -> None:
        """Test that summary includes mean total operations."""
        collector = AggregateMetricsCollector()

        collector.add_metrics(
            PerformanceMetrics(comparisons=10, swaps=5, accesses=20)
        )  # total=35
        collector.add_metrics(
            PerformanceMetrics(comparisons=20, swaps=10, accesses=30)
        )  # total=60

        summary = collector.get_summary()

        assert summary["mean_total_operations"] == 47.5  # (35+60)/2

    def test_reset(self) -> None:
        """Test resetting aggregate collector."""
        collector = AggregateMetricsCollector()

        # Add some metrics
        collector.add_metrics(PerformanceMetrics(comparisons=100))
        collector.add_metrics(PerformanceMetrics(comparisons=200))

        # Reset
        collector.reset()

        # Should be empty
        assert collector.get_all_metrics() == []
        summary = collector.get_summary()
        assert summary["runs"] == 0

    def test_get_all_metrics_returns_copy(self) -> None:
        """Test that get_all_metrics returns a copy, not the original list."""
        collector = AggregateMetricsCollector()
        metrics = PerformanceMetrics(comparisons=100)
        collector.add_metrics(metrics)

        # Get the list
        all_metrics1 = collector.get_all_metrics()

        # Modify the returned list
        all_metrics1.append(PerformanceMetrics(comparisons=200))

        # Get the list again
        all_metrics2 = collector.get_all_metrics()

        # Original should not be modified
        assert len(all_metrics2) == 1
        assert all_metrics2[0].comparisons == 100


class TestMetricsIntegration:
    """Integration tests for metrics collection with actual algorithms."""

    def test_metrics_with_simple_algorithm(self) -> None:
        """Test metrics collection with a simple sorting algorithm."""
        from complexity_profiler.algorithms.sorting import BubbleSort

        collector = DefaultMetricsCollector()
        algorithm = BubbleSort()
        data = [3, 1, 4, 1, 5, 9, 2, 6]

        with collector.timing():
            algorithm.execute(data, collector)

        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.execution_time > 0
        assert metrics.total_operations() > 0

    def test_aggregate_over_multiple_runs(self) -> None:
        """Test aggregating metrics over multiple algorithm runs."""
        from complexity_profiler.algorithms.sorting import InsertionSort

        aggregate = AggregateMetricsCollector()
        algorithm = InsertionSort()

        # Run algorithm 5 times with different data
        for size in [10, 20, 30, 40, 50]:
            collector = DefaultMetricsCollector()
            data = list(range(size, 0, -1))  # Reverse sorted

            with collector.timing():
                algorithm.execute(data, collector)

            aggregate.add_metrics(collector.get_metrics())

        summary = aggregate.get_summary()

        assert summary["runs"] == 5
        assert summary["mean_comparisons"] > 0
        assert summary["mean_execution_time"] > 0

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_metrics_scale_with_input_size(self, size: int) -> None:
        """Test that metrics scale with input size."""
        from complexity_profiler.algorithms.sorting import SelectionSort

        collector = DefaultMetricsCollector()
        algorithm = SelectionSort()
        data = list(range(size, 0, -1))

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        # For selection sort on reverse sorted data, comparisons should be ~n*(n-1)/2
        expected_comparisons = size * (size - 1) // 2
        # Allow some tolerance
        assert metrics.comparisons >= expected_comparisons * 0.9
