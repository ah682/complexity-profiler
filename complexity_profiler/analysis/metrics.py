"""
Metrics collection for algorithm execution analysis.

This module provides concrete implementations for collecting performance metrics
during algorithm execution, including operation counts, timing information, and
memory usage tracking. It implements the MetricsCollector protocol defined in
algorithms.base and provides both simple and thread-safe collector implementations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Iterator
from contextlib import contextmanager
import time
import threading


@dataclass
class PerformanceMetrics:
    """
    Container for algorithm performance metrics.

    This dataclass stores all metrics collected during algorithm execution,
    including operation counts, execution time, and memory usage statistics.

    Attributes:
        comparisons: Number of comparison operations performed
        swaps: Number of swap operations performed
        accesses: Number of array/data structure access operations
        recursive_calls: Number of recursive function calls
        execution_time: Total execution time in seconds
        memory_operations: Additional memory-related operations
        memory_usage: Peak memory usage in bytes (optional)
        custom_metrics: Dictionary for algorithm-specific custom metrics
    """

    comparisons: int = 0
    swaps: int = 0
    accesses: int = 0
    recursive_calls: int = 0
    execution_time: float = 0.0
    memory_operations: int = 0
    memory_usage: Optional[int] = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def total_operations(self) -> int:
        """
        Calculate total number of tracked operations.

        Returns:
            Sum of all operation counts (excluding execution time)
        """
        return (
            self.comparisons +
            self.swaps +
            self.accesses +
            self.recursive_calls +
            self.memory_operations
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary format.

        Returns:
            Dictionary representation of all metrics
        """
        return asdict(self)

    def add_custom_metric(self, name: str, value: Any) -> None:
        """
        Add a custom metric value.

        Args:
            name: Metric name
            value: Metric value (should be JSON-serializable)
        """
        self.custom_metrics[name] = value

    def merge(self, other: "PerformanceMetrics") -> None:
        """
        Merge another PerformanceMetrics instance into this one.

        This is useful for aggregating metrics from multiple runs or
        combining metrics from nested algorithm calls.

        Args:
            other: Another PerformanceMetrics instance to merge
        """
        self.comparisons += other.comparisons
        self.swaps += other.swaps
        self.accesses += other.accesses
        self.recursive_calls += other.recursive_calls
        self.execution_time += other.execution_time
        self.memory_operations += other.memory_operations

        if other.memory_usage is not None:
            if self.memory_usage is None:
                self.memory_usage = other.memory_usage
            else:
                self.memory_usage = max(self.memory_usage, other.memory_usage)

        # Merge custom metrics
        for key, value in other.custom_metrics.items():
            if key in self.custom_metrics:
                # If both have the same custom metric, sum them if numeric
                if isinstance(value, (int, float)) and isinstance(
                    self.custom_metrics[key], (int, float)
                ):
                    self.custom_metrics[key] += value
            else:
                self.custom_metrics[key] = value

    def __str__(self) -> str:
        """
        Format metrics as a readable string.

        Returns:
            Human-readable string representation of metrics
        """
        parts = [
            f"comparisons={self.comparisons:,}",
            f"swaps={self.swaps:,}",
            f"accesses={self.accesses:,}",
            f"recursive_calls={self.recursive_calls:,}",
            f"execution_time={self.execution_time:.6f}s",
            f"total_ops={self.total_operations():,}",
        ]
        if self.memory_usage is not None:
            parts.append(f"memory={self.memory_usage:,}b")
        if self.custom_metrics:
            custom_str = ", ".join(f"{k}={v}" for k, v in self.custom_metrics.items())
            parts.append(f"custom={{{custom_str}}}")
        return f"PerformanceMetrics({', '.join(parts)})"


class DefaultMetricsCollector:
    """
    Default implementation of the MetricsCollector protocol.

    This collector tracks all standard operations and provides timing capabilities.
    It implements the MetricsCollector protocol defined in algorithms.base and is
    suitable for single-threaded algorithm execution.

    For multi-threaded environments, use ThreadSafeMetricsCollector instead.

    Example:
        >>> collector = DefaultMetricsCollector()
        >>> with collector.timing():
        ...     collector.record_comparison()
        ...     collector.record_swap()
        >>> metrics = collector.get_metrics()
        >>> print(f"Comparisons: {metrics.comparisons}, Swaps: {metrics.swaps}")
        Comparisons: 1, Swaps: 1
    """

    def __init__(self) -> None:
        """Initialize a new metrics collector with zero counts."""
        self._metrics = PerformanceMetrics()
        self._start_time: Optional[float] = None
        self._timing_active: bool = False

    def record_comparison(self) -> None:
        """Record a comparison operation."""
        self._metrics.comparisons += 1

    def record_swap(self) -> None:
        """Record a swap operation."""
        self._metrics.swaps += 1

    def record_access(self) -> None:
        """Record an array access operation."""
        self._metrics.accesses += 1

    def record_recursive_call(self) -> None:
        """Record a recursive function call."""
        self._metrics.recursive_calls += 1

    def record_memory_operation(self) -> None:
        """Record a memory-related operation (allocation, copy, etc.)."""
        self._metrics.memory_operations += 1

    def record_custom_metric(self, name: str, value: Any) -> None:
        """
        Record a custom metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        self._metrics.add_custom_metric(name, value)

    def set_memory_usage(self, bytes_used: int) -> None:
        """
        Set the memory usage metric.

        Args:
            bytes_used: Number of bytes used
        """
        self._metrics.memory_usage = bytes_used

    def start_timing(self) -> None:
        """Start timing the execution."""
        self._start_time = time.perf_counter()
        self._timing_active = True

    def stop_timing(self) -> float:
        """
        Stop timing and record the elapsed time.

        Returns:
            The elapsed time in seconds

        Raises:
            RuntimeError: If start_timing() was not called first
        """
        if not self._timing_active or self._start_time is None:
            raise RuntimeError("start_timing() must be called before stop_timing()")

        elapsed = time.perf_counter() - self._start_time
        self._metrics.execution_time = elapsed
        self._timing_active = False
        self._start_time = None
        return elapsed

    @contextmanager
    def timing(self) -> Iterator[None]:
        """
        Context manager for automatic timing of algorithm execution.

        This is the preferred way to time algorithm execution as it
        automatically handles start and stop timing, even if an exception occurs.

        Yields:
            None

        Examples:
            >>> collector = DefaultMetricsCollector()
            >>> with collector.timing():
            ...     # Algorithm execution here
            ...     pass
            >>> print(f"Execution took {collector.get_metrics().execution_time:.4f}s")
        """
        self.start_timing()
        try:
            yield
        finally:
            try:
                self.stop_timing()
            except RuntimeError:
                self._metrics.execution_time = 0.0
                self._timing_active = False

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get the collected metrics.

        If timing is still active, it will be stopped automatically.

        Returns:
            PerformanceMetrics object with all collected data
        """
        if self._timing_active:
            try:
                self.stop_timing()
            except RuntimeError:
                self._metrics.execution_time = 0.0
                self._timing_active = False
        return self._metrics

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._metrics = PerformanceMetrics()
        self._start_time = None
        self._timing_active = False

    def get_execution_time(self) -> float:
        """
        Get the current execution time.

        Returns:
            Execution time in seconds
        """
        if self._timing_active and self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._metrics.execution_time

    def __str__(self) -> str:
        """Return string representation of current metrics."""
        return str(self._metrics)

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"DefaultMetricsCollector(metrics={self._metrics!r})"


class ThreadSafeMetricsCollector:
    """
    Thread-safe implementation of the MetricsCollector protocol.

    This class provides a thread-safe way to collect metrics during algorithm
    execution. It can be safely used in multi-threaded environments where
    multiple threads may be recording metrics concurrently.

    The collector uses a lock to ensure thread-safety for all operations,
    making it safe to use from multiple threads without external synchronization.

    Example:
        >>> collector = ThreadSafeMetricsCollector()
        >>> with collector.timing():
        ...     collector.record_comparison()
        ...     collector.record_swap()
        >>> metrics = collector.get_metrics()
        >>> print(f"Comparisons: {metrics.comparisons}")
    """

    def __init__(self) -> None:
        """Initialize the collector with zero metrics and a lock."""
        self._metrics = PerformanceMetrics()
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    @property
    def metrics(self) -> PerformanceMetrics:
        """
        Get the current metrics (thread-safe).

        Returns:
            Copy of current metrics to prevent external modification
        """
        with self._lock:
            # Return a copy to prevent external modification
            return PerformanceMetrics(
                comparisons=self._metrics.comparisons,
                swaps=self._metrics.swaps,
                accesses=self._metrics.accesses,
                recursive_calls=self._metrics.recursive_calls,
                execution_time=self._metrics.execution_time,
                memory_operations=self._metrics.memory_operations,
                memory_usage=self._metrics.memory_usage,
                custom_metrics=self._metrics.custom_metrics.copy(),
            )

    def record_comparison(self) -> None:
        """Record a comparison operation (thread-safe)."""
        with self._lock:
            self._metrics.comparisons += 1

    def record_swap(self) -> None:
        """Record a swap operation (thread-safe)."""
        with self._lock:
            self._metrics.swaps += 1

    def record_access(self) -> None:
        """Record an array access operation (thread-safe)."""
        with self._lock:
            self._metrics.accesses += 1

    def record_recursive_call(self) -> None:
        """Record a recursive function call (thread-safe)."""
        with self._lock:
            self._metrics.recursive_calls += 1

    def record_memory_operation(self) -> None:
        """Record a memory-related operation (thread-safe)."""
        with self._lock:
            self._metrics.memory_operations += 1

    def record_custom_metric(self, name: str, value: Any) -> None:
        """
        Record a custom metric value (thread-safe).

        Args:
            name: Metric name
            value: Metric value
        """
        with self._lock:
            self._metrics.add_custom_metric(name, value)

    def set_memory_usage(self, bytes_used: int) -> None:
        """
        Set the memory usage metric (thread-safe).

        Args:
            bytes_used: Number of bytes used
        """
        with self._lock:
            self._metrics.memory_usage = bytes_used

    def start_timing(self) -> None:
        """Start timing the algorithm execution."""
        with self._lock:
            self._start_time = time.perf_counter()

    def stop_timing(self) -> float:
        """
        Stop timing and record the execution time.

        Returns:
            The elapsed time in seconds

        Raises:
            RuntimeError: If start_timing() was not called first
        """
        with self._lock:
            if self._start_time is None:
                raise RuntimeError("start_timing() must be called before stop_timing()")

            elapsed = time.perf_counter() - self._start_time
            self._metrics.execution_time = elapsed
            self._start_time = None
            return elapsed

    @contextmanager
    def timing(self) -> Iterator[None]:
        """
        Context manager for automatic timing of algorithm execution.

        Yields:
            None

        Examples:
            >>> collector = ThreadSafeMetricsCollector()
            >>> with collector.timing():
            ...     # Algorithm execution here
            ...     pass
        """
        self.start_timing()
        try:
            yield
        finally:
            try:
                self.stop_timing()
            except RuntimeError:
                with self._lock:
                    self._metrics.execution_time = 0.0

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get the collected metrics.

        This method is part of the MetricsCollector protocol.

        Returns:
            Current metrics snapshot
        """
        return self.metrics

    def reset(self) -> None:
        """Reset all metrics to initial state (thread-safe)."""
        with self._lock:
            self._metrics = PerformanceMetrics()
            self._start_time = None

    def __str__(self) -> str:
        """Return string representation of current metrics."""
        return str(self.metrics)

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"ThreadSafeMetricsCollector(metrics={self.metrics!r})"


class AggregateMetricsCollector:
    """
    Collector for aggregating metrics across multiple runs.

    Useful for collecting statistics over multiple executions of the same algorithm.
    This class is particularly useful for profiling with multiple runs to compute
    average performance characteristics.

    Example:
        >>> aggregate = AggregateMetricsCollector()
        >>> for _ in range(3):
        ...     collector = DefaultMetricsCollector()
        ...     collector.record_comparison()
        ...     aggregate.add_metrics(collector.get_metrics())
        >>> summary = aggregate.get_summary()
        >>> print(f"Total runs: {summary['runs']}")
        Total runs: 3
    """

    def __init__(self) -> None:
        """Initialize aggregate collector."""
        self._all_metrics: list[PerformanceMetrics] = []

    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        Add metrics from a single run.

        Args:
            metrics: PerformanceMetrics to add to the aggregate
        """
        self._all_metrics.append(metrics)

    def get_all_metrics(self) -> list[PerformanceMetrics]:
        """
        Get all collected metrics.

        Returns:
            List of all PerformanceMetrics objects
        """
        return self._all_metrics.copy()

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all runs.

        Returns:
            Dictionary containing mean values for each metric and run count
        """
        if not self._all_metrics:
            return {
                'runs': 0,
                'mean_comparisons': 0.0,
                'mean_swaps': 0.0,
                'mean_accesses': 0.0,
                'mean_recursive_calls': 0.0,
                'mean_execution_time': 0.0,
                'mean_total_operations': 0.0,
                'mean_memory_operations': 0.0,
            }

        num_runs = len(self._all_metrics)
        return {
            'runs': num_runs,
            'mean_comparisons': sum(m.comparisons for m in self._all_metrics) / num_runs,
            'mean_swaps': sum(m.swaps for m in self._all_metrics) / num_runs,
            'mean_accesses': sum(m.accesses for m in self._all_metrics) / num_runs,
            'mean_recursive_calls': sum(m.recursive_calls for m in self._all_metrics) / num_runs,
            'mean_execution_time': sum(m.execution_time for m in self._all_metrics) / num_runs,
            'mean_total_operations': sum(m.total_operations() for m in self._all_metrics) / num_runs,
            'mean_memory_operations': sum(m.memory_operations for m in self._all_metrics) / num_runs,
        }

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._all_metrics.clear()


@contextmanager
def measure_time() -> Iterator[list[float]]:
    """
    Context manager for simple time measurement.

    This is a lightweight utility for measuring execution time without
    the full metrics collection infrastructure.

    Yields:
        A list with a single element that will be populated with elapsed time

    Examples:
        >>> elapsed = []
        >>> with measure_time() as elapsed:
        ...     # Code to measure
        ...     pass
        >>> print(f"Elapsed: {elapsed[0]:.4f}s")
    """
    start = time.perf_counter()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        elapsed[0] = time.perf_counter() - start


__all__ = [
    "PerformanceMetrics",
    "DefaultMetricsCollector",
    "ThreadSafeMetricsCollector",
    "AggregateMetricsCollector",
    "measure_time",
]
