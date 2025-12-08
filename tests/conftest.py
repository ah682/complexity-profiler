"""
Pytest configuration and shared fixtures for Big-O Complexity Analyzer tests.

This module provides common fixtures and test utilities used across all test modules,
including sample data generators, metrics collectors, and profiler instances.
"""

from typing import Callable, Any
import random
import pytest
import numpy as np

from complexity_profiler.analysis.metrics import DefaultMetricsCollector, PerformanceMetrics
from complexity_profiler.algorithms.sorting import (
    MergeSort,
    QuickSort,
    SelectionSort,
    BubbleSort,
    InsertionSort,
    HeapSort,
)


@pytest.fixture
def sample_data() -> dict[str, list[int]]:
    """
    Provide various sample data configurations for testing algorithms.

    Returns:
        Dictionary containing different data configurations:
        - random: Randomly shuffled integers
        - sorted: Already sorted ascending
        - reverse: Sorted in descending order
        - nearly_sorted: Mostly sorted with few out-of-place elements
        - duplicates: Contains many duplicate values
        - single: Single element
        - empty: Empty list

    Example:
        >>> def test_sorting(sample_data):
        ...     data = sample_data['random']
        ...     assert len(data) == 100
    """
    size = 100
    base_data = list(range(size))

    # Create random data
    random_data = base_data.copy()
    random.shuffle(random_data)

    # Create nearly sorted data (swap 5% of elements)
    nearly_sorted = base_data.copy()
    num_swaps = max(1, size // 20)
    for _ in range(num_swaps):
        i, j = random.randint(0, size - 1), random.randint(0, size - 1)
        nearly_sorted[i], nearly_sorted[j] = nearly_sorted[j], nearly_sorted[i]

    # Create data with duplicates
    duplicates = [random.randint(0, 20) for _ in range(size)]

    return {
        "random": random_data,
        "sorted": base_data.copy(),
        "reverse": sorted(base_data, reverse=True),
        "nearly_sorted": nearly_sorted,
        "duplicates": duplicates,
        "single": [42],
        "empty": [],
    }


@pytest.fixture
def small_sample_data() -> dict[str, list[int]]:
    """
    Provide small sample data for quick tests.

    Returns:
        Dictionary with small data configurations for fast test execution.

    Example:
        >>> def test_quick(small_sample_data):
        ...     data = small_sample_data['random']
        ...     assert len(data) <= 20
    """
    size = 20
    base_data = list(range(size))

    random_data = base_data.copy()
    random.shuffle(random_data)

    return {
        "random": random_data,
        "sorted": base_data.copy(),
        "reverse": sorted(base_data, reverse=True),
        "single": [42],
        "empty": [],
    }


@pytest.fixture
def metrics_collector() -> DefaultMetricsCollector:
    """
    Provide a fresh metrics collector for each test.

    Returns:
        New DefaultMetricsCollector instance with zero counts.

    Example:
        >>> def test_metrics(metrics_collector):
        ...     metrics_collector.record_comparison()
        ...     assert metrics_collector.get_metrics().comparisons == 1
    """
    return DefaultMetricsCollector()


@pytest.fixture
def sorting_algorithms() -> dict[str, Any]:
    """
    Provide instances of all sorting algorithms.

    Returns:
        Dictionary mapping algorithm names to instances.

    Example:
        >>> def test_all_sorts(sorting_algorithms):
        ...     for name, algo in sorting_algorithms.items():
        ...         assert hasattr(algo, 'execute')
    """
    return {
        "MergeSort": MergeSort(),
        "QuickSort": QuickSort(),
        "SelectionSort": SelectionSort(),
        "BubbleSort": BubbleSort(),
        "InsertionSort": InsertionSort(),
        "HeapSort": HeapSort(),
    }


@pytest.fixture
def performance_data_generator() -> Callable[[str, int], tuple[np.ndarray, np.ndarray]]:
    """
    Provide a function to generate synthetic performance data for complexity testing.

    Returns:
        Function that generates (sizes, times) arrays for a given complexity.

    Example:
        >>> def test_complexity(performance_data_generator):
        ...     sizes, times = performance_data_generator('linear', 10)
        ...     assert len(sizes) == 10
        ...     assert len(times) == 10
    """

    def generator(complexity: str, num_points: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic performance data matching a specific complexity.

        Args:
            complexity: One of 'constant', 'logarithmic', 'linear',
                       'linearithmic', 'quadratic', 'cubic'
            num_points: Number of data points to generate

        Returns:
            Tuple of (sizes, times) arrays

        Raises:
            ValueError: If complexity is not recognized
        """
        # Generate input sizes
        sizes = np.linspace(10, 1000, num_points)

        # Generate times based on complexity with small random noise
        noise_factor = 0.05  # 5% noise

        if complexity == "constant":
            base_times = np.ones_like(sizes) * 0.001
        elif complexity == "logarithmic":
            base_times = 0.0001 * np.log(sizes)
        elif complexity == "linear":
            base_times = 0.00001 * sizes
        elif complexity == "linearithmic":
            base_times = 0.000001 * sizes * np.log(sizes)
        elif complexity == "quadratic":
            base_times = 0.0000001 * sizes**2
        elif complexity == "cubic":
            base_times = 0.00000001 * sizes**3
        else:
            raise ValueError(f"Unknown complexity: {complexity}")

        # Add small random noise
        noise = np.random.normal(1.0, noise_factor, num_points)
        times = base_times * noise

        # Ensure no negative times
        times = np.abs(times)

        return sizes, times

    return generator


@pytest.fixture
def is_sorted() -> Callable[[list[Any]], bool]:
    """
    Provide a helper function to check if a list is sorted.

    Returns:
        Function that returns True if list is sorted in ascending order.

    Example:
        >>> def test_sorting(is_sorted):
        ...     assert is_sorted([1, 2, 3, 4, 5])
        ...     assert not is_sorted([1, 3, 2, 4, 5])
    """

    def check(data: list[Any]) -> bool:
        """Check if list is sorted in ascending order."""
        return all(data[i] <= data[i + 1] for i in range(len(data) - 1))

    return check


@pytest.fixture
def assert_sorted() -> Callable[[list[Any]], None]:
    """
    Provide a helper function to assert that a list is sorted.

    Returns:
        Function that raises AssertionError if list is not sorted.

    Example:
        >>> def test_sorting(assert_sorted):
        ...     assert_sorted([1, 2, 3, 4, 5])
        ...     # assert_sorted([1, 3, 2])  # Would raise AssertionError
    """

    def check(data: list[Any]) -> None:
        """Assert that list is sorted in ascending order."""
        for i in range(len(data) - 1):
            if data[i] > data[i + 1]:
                raise AssertionError(
                    f"List not sorted at index {i}: {data[i]} > {data[i + 1]}"
                )

    return check


@pytest.fixture
def performance_metrics_factory() -> Callable[..., PerformanceMetrics]:
    """
    Provide a factory function for creating PerformanceMetrics instances.

    Returns:
        Function that creates PerformanceMetrics with specified values.

    Example:
        >>> def test_metrics(performance_metrics_factory):
        ...     metrics = performance_metrics_factory(comparisons=100, swaps=50)
        ...     assert metrics.comparisons == 100
    """

    def factory(**kwargs: Any) -> PerformanceMetrics:
        """
        Create PerformanceMetrics with custom values.

        Args:
            **kwargs: Keyword arguments for PerformanceMetrics fields

        Returns:
            PerformanceMetrics instance
        """
        return PerformanceMetrics(**kwargs)

    return factory


# Pytest configuration
def pytest_configure(config: Any) -> None:
    """
    Configure pytest with custom markers.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "property: marks tests as property-based tests"
    )
