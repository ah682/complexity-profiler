"""
Algorithm profiling and performance analysis tools.

This module provides tools for profiling algorithms, collecting performance
metrics across multiple runs with different input sizes, and analyzing the
results to determine empirical Big-O complexity.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
import copy
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from complexity_profiler.algorithms.base import ComplexityClass, Algorithm
from complexity_profiler.analysis.metrics import PerformanceMetrics, DefaultMetricsCollector
from complexity_profiler.analysis.statistics import Statistics, compute_statistics
from complexity_profiler.analysis.curve_fitting import ComplexityFitter


@dataclass
class ProfileResult:
    """
    Complete profiling results for an algorithm.

    This dataclass encapsulates all information collected during algorithm
    profiling, including metadata, measured metrics, and statistical analysis.

    Attributes:
        algorithm_name: Name of the profiled algorithm
        category: Category of the algorithm (e.g., "sorting", "searching")
        input_sizes: List of input sizes used in profiling
        metrics_per_size: Dictionary mapping input size to collected metrics
        statistics_per_size: Dictionary mapping input size to statistics
        empirical_complexity: Detected Big-O complexity class (e.g., "O(n)")
        r_squared: R-squared value of the fitted curve (0.0 to 1.0)
        timestamp: When the profiling was performed
        notes: Optional notes about the profiling session
    """

    algorithm_name: str
    category: str
    input_sizes: list[int]
    metrics_per_size: dict[int, list[PerformanceMetrics]]
    statistics_per_size: dict[int, Statistics]
    empirical_complexity: str
    r_squared: float
    timestamp: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the profile result after initialization."""
        if not self.input_sizes:
            raise ValueError("input_sizes cannot be empty")
        if not self.algorithm_name:
            raise ValueError("algorithm_name cannot be empty")
        if not (0 <= self.r_squared <= 1):
            raise ValueError("r_squared must be between 0 and 1")

    def to_dict(self) -> dict:
        """
        Convert the ProfileResult to a dictionary, handling dataclass conversion.

        This method recursively converts nested dataclasses (PerformanceMetrics
        and Statistics) to dictionaries for serialization.

        Returns:
            Dictionary representation of the ProfileResult
        """
        result = asdict(self)

        # Convert timestamp to ISO format string
        if isinstance(result['timestamp'], datetime):
            result['timestamp'] = result['timestamp'].isoformat()

        # Convert metrics to dictionaries
        metrics_dict = {}
        for size, metrics_list in self.metrics_per_size.items():
            metrics_dict[str(size)] = [asdict(m) for m in metrics_list]
        result['metrics_per_size'] = metrics_dict

        # Convert statistics to dictionaries
        stats_dict = {}
        for size, stats in self.statistics_per_size.items():
            stats_dict[str(size)] = asdict(stats)
        result['statistics_per_size'] = stats_dict

        return result

    def get_summary(self) -> dict:
        """
        Get a summary of the profiling results.

        Returns a high-level overview of the profiling results including
        algorithm information, detected complexity, and fit quality.

        Returns:
            Dictionary with summary information
        """
        return {
            'algorithm_name': self.algorithm_name,
            'category': self.category,
            'input_sizes': self.input_sizes,
            'empirical_complexity': self.empirical_complexity,
            'r_squared': self.r_squared,
            'timestamp': self.timestamp.isoformat(),
            'notes': self.notes,
        }


class AlgorithmProfiler:
    """
    Profiles algorithms to determine empirical complexity.

    Uses a fluent API for configuration and performs comprehensive profiling
    including warmup runs, multiple executions per size, and statistical analysis.

    Example:
        >>> profiler = AlgorithmProfiler()
        >>> result = (profiler
        ...     .with_sizes([10, 50, 100, 500, 1000])
        ...     .with_runs(5)
        ...     .with_data_generator(lambda n: list(range(n)))
        ...     .profile(my_algorithm))
        >>> print(f"Detected complexity: {result.empirical_complexity}")
        Detected complexity: O(n log n)
    """

    def __init__(
        self,
        warmup_runs: int = 2,
        min_r_squared: float = 0.8,
    ) -> None:
        """
        Initialize the algorithm profiler.

        Args:
            warmup_runs: Number of warmup runs before actual profiling (default: 2)
            min_r_squared: Minimum R² for acceptable complexity fit (default: 0.8)
        """
        self._input_sizes: list[int] = [10, 50, 100, 500, 1000]
        self._runs_per_size: int = 3
        self._data_generator: Optional[Callable[[int], list[Any]]] = None
        self._warmup_runs = warmup_runs
        self._min_r_squared = min_r_squared

    def with_sizes(self, sizes: list[int]) -> 'AlgorithmProfiler':
        """
        Set the input sizes to test.

        Args:
            sizes: List of input sizes to profile

        Returns:
            Self for method chaining

        Raises:
            ValueError: If sizes list is empty or contains non-positive values
        """
        if not sizes:
            raise ValueError("sizes list cannot be empty")
        if any(s <= 0 for s in sizes):
            raise ValueError("all sizes must be positive")

        self._input_sizes = sorted(sizes)
        return self

    def with_runs(self, runs: int) -> 'AlgorithmProfiler':
        """
        Set the number of runs per input size.

        Args:
            runs: Number of times to run the algorithm for each size

        Returns:
            Self for method chaining

        Raises:
            ValueError: If runs is not positive
        """
        if runs <= 0:
            raise ValueError("runs must be positive")

        self._runs_per_size = runs
        return self

    def with_data_generator(
        self,
        generator: Callable[[int], list[Any]]
    ) -> 'AlgorithmProfiler':
        """
        Set the data generator function.

        Args:
            generator: Function that takes a size and returns test data

        Returns:
            Self for method chaining
        """
        self._data_generator = generator
        return self

    def profile(self, algorithm: Algorithm) -> ProfileResult:
        """
        Profile the algorithm and return comprehensive results.

        Performs the following steps:
        1. Warmup runs to prime caches
        2. Multiple profiling runs for each input size
        3. Metrics collection and statistical analysis
        4. Complexity curve fitting

        Args:
            algorithm: The algorithm to profile

        Returns:
            ProfileResult containing all profiling data and analysis

        Raises:
            ValueError: If data generator is not set
            RuntimeError: If profiling fails or complexity cannot be determined
        """
        if self._data_generator is None:
            raise ValueError("Data generator must be set before profiling")

        # Collect metrics for all sizes
        metrics_by_size: dict[int, list[PerformanceMetrics]] = {}
        statistics_by_size: dict[int, Statistics] = {}

        for size in self._input_sizes:
            metrics_list = self._profile_single_size(algorithm, size)
            metrics_by_size[size] = metrics_list

            # Compute statistics from execution times
            times = [m.execution_time for m in metrics_list]
            statistics_by_size[size] = compute_statistics(times)

        # Fit complexity curve
        sizes_arr = np.array(self._input_sizes, dtype=np.float64)
        mean_times = np.array(
            [statistics_by_size[size].mean for size in self._input_sizes],
            dtype=np.float64
        )

        fitter = ComplexityFitter(min_r_squared=self._min_r_squared)
        fitted_complexity, fit_quality = fitter.fit_complexity(sizes_arr, mean_times)

        return ProfileResult(
            algorithm_name=algorithm.metadata.name,
            category=algorithm.metadata.category,
            input_sizes=self._input_sizes,
            metrics_per_size=metrics_by_size,
            statistics_per_size=statistics_by_size,
            empirical_complexity=str(fitted_complexity),
            r_squared=fit_quality,
        )

    def _profile_single_size(
        self,
        algorithm: Algorithm,
        size: int,
    ) -> list[PerformanceMetrics]:
        """
        Profile algorithm for a single input size.

        Performs warmup runs followed by actual profiling runs.

        Args:
            algorithm: The algorithm to profile
            size: Input size to test

        Returns:
            List of PerformanceMetrics from each run
        """
        # Generate test data once
        test_data = self._data_generator(size)  # type: ignore

        # Warmup runs (not recorded)
        for _ in range(self._warmup_runs):
            collector = DefaultMetricsCollector()
            data_copy = copy.deepcopy(test_data)
            collector.start_timing()
            algorithm.execute(data_copy, collector)
            collector.stop_timing()

        # Actual profiling runs
        metrics_list: list[PerformanceMetrics] = []

        for _ in range(self._runs_per_size):
            collector = DefaultMetricsCollector()
            data_copy = copy.deepcopy(test_data)

            collector.start_timing()
            algorithm.execute(data_copy, collector)
            collector.stop_timing()

            metrics_list.append(collector.get_metrics())

        return metrics_list


def profile_algorithm(
    algorithm: Algorithm,
    sizes: list[int],
    runs: int = 3,
    data_generator: Optional[Callable[[int], list[Any]]] = None,
) -> ProfileResult:
    """
    Convenience function to profile an algorithm.

    This is a simplified interface to AlgorithmProfiler for quick profiling.

    Args:
        algorithm: The algorithm to profile
        sizes: List of input sizes to test
        runs: Number of runs per size (default: 3)
        data_generator: Function to generate test data (default: random integers)

    Returns:
        ProfileResult with comprehensive profiling data

    Example:
        >>> def my_generator(n):
        ...     return list(range(n, 0, -1))
        >>> result = profile_algorithm(bubble_sort, [10, 50, 100], runs=5,
        ...                            data_generator=my_generator)
        >>> print(result.empirical_complexity)
        O(n²)
    """
    if data_generator is None:
        # Default generator: random integers
        import random
        data_generator = lambda n: [random.randint(0, 1000) for _ in range(n)]

    profiler = AlgorithmProfiler()
    return (profiler
            .with_sizes(sizes)
            .with_runs(runs)
            .with_data_generator(data_generator)
            .profile(algorithm))


__all__ = [
    "ProfileResult",
    "AlgorithmProfiler",
    "profile_algorithm",
]
