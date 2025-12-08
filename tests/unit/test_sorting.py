"""
Unit tests for sorting algorithm implementations.

This module provides comprehensive tests for all sorting algorithms including:
- Correctness verification on various input types
- Algorithm-specific property tests
- Edge case handling
- Metrics collection accuracy
- Performance characteristics
"""

import pytest
import random
from typing import Any, Type, Callable

from complexity_profiler.algorithms.sorting import (
    MergeSort,
    QuickSort,
    SelectionSort,
    BubbleSort,
    InsertionSort,
    HeapSort,
)
from complexity_profiler.algorithms.base import Algorithm, ComplexityClass
from complexity_profiler.analysis.metrics import DefaultMetricsCollector
from complexity_profiler.data.generators import (
    random_data,
    sorted_data,
    reverse_sorted_data,
    nearly_sorted_data,
    duplicates_data,
    uniform_data,
)


# Parametrize all sorting algorithms for common tests
ALL_SORTING_ALGORITHMS = [
    (MergeSort, "MergeSort"),
    (QuickSort, "QuickSort"),
    (SelectionSort, "SelectionSort"),
    (BubbleSort, "BubbleSort"),
    (InsertionSort, "InsertionSort"),
    (HeapSort, "HeapSort"),
]


class TestSortingCorrectness:
    """Test suite for verifying sorting correctness across all algorithms."""

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_random_data(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test that algorithm correctly sorts random data."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = random_data(100, seed=42)

        result = algorithm.execute(data, collector)

        assert len(result) == len(data)
        assert sorted(data) == result
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_already_sorted_data(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting already sorted data (best case for some algorithms)."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = sorted_data(50)

        result = algorithm.execute(data, collector)

        assert result == data
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_reverse_sorted_data(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting reverse sorted data (worst case for many algorithms)."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = reverse_sorted_data(50)

        result = algorithm.execute(data, collector)

        expected = list(range(50))
        assert result == expected
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_data_with_duplicates(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting data containing many duplicate values."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = duplicates_data(100, num_unique=10, seed=42)

        result = algorithm.execute(data, collector)

        assert len(result) == len(data)
        assert sorted(data) == result
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_uniform_data(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting data where all elements are identical."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = uniform_data(50, value=42)

        result = algorithm.execute(data, collector)

        assert result == data
        assert all(x == 42 for x in result)

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_nearly_sorted_data(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting nearly sorted data (good case for adaptive algorithms)."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = nearly_sorted_data(100, swaps=5, seed=42)

        result = algorithm.execute(data, collector)

        assert sorted(data) == result
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_single_element(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting array with single element."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = [42]

        result = algorithm.execute(data, collector)

        assert result == [42]

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_empty_array(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting empty array."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = []

        result = algorithm.execute(data, collector)

        assert result == []

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_two_elements(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting array with two elements."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()

        # Test both orderings
        result1 = algorithm.execute([2, 1], DefaultMetricsCollector())
        result2 = algorithm.execute([1, 2], DefaultMetricsCollector())

        assert result1 == [1, 2]
        assert result2 == [1, 2]

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_sorts_negative_numbers(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting array with negative numbers."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = [5, -3, 8, -1, 0, 2, -7]

        result = algorithm.execute(data, collector)

        assert result == [-7, -3, -1, 0, 2, 5, 8]

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_does_not_modify_input(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test that algorithm doesn't modify the input array."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        original_data = [3, 1, 4, 1, 5, 9, 2, 6]
        data = original_data.copy()

        result = algorithm.execute(data, collector)

        # Input should be unchanged (algorithms use copy)
        assert data == original_data
        # Result should be sorted
        assert result == sorted(original_data)


class TestSortingStability:
    """Test suite for verifying stability of stable sorting algorithms."""

    @pytest.fixture
    def stable_data(self) -> list[tuple[int, str]]:
        """
        Create test data for stability testing.

        Returns tuples of (key, label) where stability can be verified
        by checking if equal keys maintain their original order.
        """
        return [
            (3, 'a'), (1, 'b'), (3, 'c'), (2, 'd'),
            (1, 'e'), (3, 'f'), (2, 'g'), (1, 'h')
        ]

    @pytest.mark.parametrize("algorithm_class,expected_stable", [
        (MergeSort, True),
        (BubbleSort, True),
        (InsertionSort, True),
        (QuickSort, False),
        (SelectionSort, False),
        (HeapSort, False),
    ])
    def test_stability(
        self,
        algorithm_class: Type[Algorithm],
        expected_stable: bool,
        stable_data: list[tuple[int, str]]
    ) -> None:
        """Test whether algorithm maintains stability for equal elements."""
        # Create a custom comparable class that tracks original position
        class TrackedValue:
            def __init__(self, key: int, label: str):
                self.key = key
                self.label = label

            def __lt__(self, other: Any) -> bool:
                return self.key < other.key

            def __le__(self, other: Any) -> bool:
                return self.key <= other.key

            def __gt__(self, other: Any) -> bool:
                return self.key > other.key

            def __ge__(self, other: Any) -> bool:
                return self.key >= other.key

            def __eq__(self, other: Any) -> bool:
                return self.key == other.key

        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()

        # Convert to tracked values
        data = [TrackedValue(k, l) for k, l in stable_data]
        result = algorithm.execute(data, collector)

        # Check if equal elements maintained their relative order
        # Group by key
        groups = {}
        for item in result:
            if item.key not in groups:
                groups[item.key] = []
            groups[item.key].append(item.label)

        # For stable sorts, equal keys should maintain original order
        if expected_stable:
            # Key 1: should be [b, e, h]
            # Key 2: should be [d, g]
            # Key 3: should be [a, c, f]
            assert groups[1] == ['b', 'e', 'h'], f"{algorithm_class.__name__} failed stability test"
            assert groups[2] == ['d', 'g'], f"{algorithm_class.__name__} failed stability test"
            assert groups[3] == ['a', 'c', 'f'], f"{algorithm_class.__name__} failed stability test"


class TestSortingMetrics:
    """Test suite for verifying metrics collection during sorting."""

    def test_merge_sort_records_metrics(self) -> None:
        """Test that MergeSort records comparisons and operations."""
        algorithm = MergeSort()
        collector = DefaultMetricsCollector()
        data = random_data(50, seed=42)

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.recursive_calls > 0
        assert metrics.accesses > 0
        assert metrics.swaps > 0  # Merging counts as swaps

    def test_quick_sort_records_metrics(self) -> None:
        """Test that QuickSort records comparisons and partitioning operations."""
        algorithm = QuickSort()
        collector = DefaultMetricsCollector()
        data = random_data(50, seed=42)

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.recursive_calls > 0
        assert metrics.accesses > 0
        assert metrics.swaps > 0

    def test_selection_sort_records_metrics(self) -> None:
        """Test that SelectionSort records comparisons and swaps."""
        algorithm = SelectionSort()
        collector = DefaultMetricsCollector()
        data = [5, 2, 8, 1, 9]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        # Selection sort always does n*(n-1)/2 comparisons
        n = len(data)
        expected_comparisons = n * (n - 1) // 2
        assert metrics.comparisons == expected_comparisons
        assert metrics.swaps > 0

    def test_bubble_sort_records_metrics(self) -> None:
        """Test that BubbleSort records comparisons and swaps."""
        algorithm = BubbleSort()
        collector = DefaultMetricsCollector()
        data = [3, 1, 4, 1, 5]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0
        assert metrics.accesses > 0

    def test_insertion_sort_records_metrics(self) -> None:
        """Test that InsertionSort records comparisons and swaps."""
        algorithm = InsertionSort()
        collector = DefaultMetricsCollector()
        data = reverse_sorted_data(20)

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0
        assert metrics.accesses > 0

    def test_heap_sort_records_metrics(self) -> None:
        """Test that HeapSort records comparisons and swaps."""
        algorithm = HeapSort()
        collector = DefaultMetricsCollector()
        data = random_data(30, seed=42)

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0
        assert metrics.accesses > 0
        assert metrics.recursive_calls > 0

    @pytest.mark.parametrize("size", [10, 20, 50])
    def test_metrics_scale_with_input_size(self, size: int) -> None:
        """Test that operation counts scale appropriately with input size."""
        algorithm = MergeSort()

        data = random_data(size, seed=42)
        collector = DefaultMetricsCollector()
        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        # For MergeSort, comparisons should be O(n log n)
        # Rough approximation: should be at least n and at most n^2
        assert metrics.comparisons >= size
        assert metrics.comparisons <= size * size


class TestSortingAlgorithmMetadata:
    """Test suite for verifying algorithm metadata."""

    def test_merge_sort_metadata(self) -> None:
        """Test MergeSort metadata is correct."""
        algorithm = MergeSort()
        metadata = algorithm.metadata

        assert metadata.name == "Merge Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.LINEARITHMIC
        assert metadata.stable is True
        assert metadata.in_place is False
        assert "divide-and-conquer" in metadata.description.lower()

    def test_quick_sort_metadata(self) -> None:
        """Test QuickSort metadata is correct."""
        algorithm = QuickSort()
        metadata = algorithm.metadata

        assert metadata.name == "Quick Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.LINEARITHMIC
        assert metadata.stable is False
        assert "partition" in metadata.description.lower()

    def test_selection_sort_metadata(self) -> None:
        """Test SelectionSort metadata is correct."""
        algorithm = SelectionSort()
        metadata = algorithm.metadata

        assert metadata.name == "Selection Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.QUADRATIC
        assert metadata.stable is False
        assert metadata.in_place is True

    def test_bubble_sort_metadata(self) -> None:
        """Test BubbleSort metadata is correct."""
        algorithm = BubbleSort()
        metadata = algorithm.metadata

        assert metadata.name == "Bubble Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.QUADRATIC
        assert metadata.stable is True
        assert metadata.in_place is True
        assert "bubble" in metadata.description.lower()

    def test_insertion_sort_metadata(self) -> None:
        """Test InsertionSort metadata is correct."""
        algorithm = InsertionSort()
        metadata = algorithm.metadata

        assert metadata.name == "Insertion Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.QUADRATIC
        assert metadata.stable is True
        assert metadata.in_place is True
        assert "adaptive" in metadata.description.lower()

    def test_heap_sort_metadata(self) -> None:
        """Test HeapSort metadata is correct."""
        algorithm = HeapSort()
        metadata = algorithm.metadata

        assert metadata.name == "Heap Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.LINEARITHMIC
        assert metadata.stable is False
        assert metadata.in_place is True
        assert "heap" in metadata.description.lower()


class TestSortingEdgeCases:
    """Test suite for edge cases and special scenarios."""

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_large_dataset(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting large dataset (performance test)."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()

        # Use smaller size for quadratic algorithms
        if algorithm_class in [SelectionSort, BubbleSort, InsertionSort]:
            data = random_data(500, seed=42)
        else:
            data = random_data(1000, seed=42)

        result = algorithm.execute(data, collector)

        assert len(result) == len(data)
        assert result == sorted(data)

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_alternating_values(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting alternating high/low values."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = [1, 100, 2, 99, 3, 98, 4, 97, 5, 96]

        result = algorithm.execute(data, collector)

        assert result == sorted(data)

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_power_of_two_sizes(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test sorting arrays with power-of-two sizes."""
        algorithm = algorithm_class()

        for size in [2, 4, 8, 16, 32, 64]:
            collector = DefaultMetricsCollector()
            data = random_data(size, seed=42)
            result = algorithm.execute(data, collector)
            assert result == sorted(data)


class TestAdaptiveAlgorithms:
    """Test suite specifically for adaptive algorithm behavior."""

    def test_bubble_sort_early_termination(self) -> None:
        """Test that BubbleSort terminates early on sorted data."""
        algorithm = BubbleSort()
        collector_sorted = DefaultMetricsCollector()
        collector_random = DefaultMetricsCollector()

        # Sorted data should require fewer operations
        sorted_input = sorted_data(100)
        random_input = random_data(100, seed=42)

        algorithm.execute(sorted_input, collector_sorted)
        algorithm.execute(random_input, collector_random)

        metrics_sorted = collector_sorted.get_metrics()
        metrics_random = collector_random.get_metrics()

        # Sorted should have significantly fewer comparisons
        assert metrics_sorted.comparisons < metrics_random.comparisons
        assert metrics_sorted.swaps == 0  # No swaps needed

    def test_insertion_sort_efficiency_on_nearly_sorted(self) -> None:
        """Test that InsertionSort is efficient on nearly sorted data."""
        algorithm = InsertionSort()
        collector_nearly = DefaultMetricsCollector()
        collector_random = DefaultMetricsCollector()

        nearly_sorted_input = nearly_sorted_data(100, swaps=5, seed=42)
        random_input = random_data(100, seed=42)

        algorithm.execute(nearly_sorted_input, collector_nearly)
        algorithm.execute(random_input, collector_random)

        metrics_nearly = collector_nearly.get_metrics()
        metrics_random = collector_random.get_metrics()

        # Nearly sorted should have fewer operations
        assert metrics_nearly.comparisons < metrics_random.comparisons


class TestSortingPropertyBased:
    """Property-based tests for sorting algorithms."""

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 12345])
    def test_idempotent(
        self, algorithm_class: Type[Algorithm], name: str, seed: int
    ) -> None:
        """Test that sorting twice produces same result as sorting once."""
        algorithm = algorithm_class()
        data = random_data(50, seed=seed)

        result1 = algorithm.execute(data, DefaultMetricsCollector())
        result2 = algorithm.execute(result1, DefaultMetricsCollector())

        assert result1 == result2

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_permutation_invariance(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test that result contains same elements as input (permutation)."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = random_data(100, seed=42)

        result = algorithm.execute(data, collector)

        # Check same length
        assert len(result) == len(data)

        # Check same elements (sorted to compare)
        assert sorted(result) == sorted(data)

        # Check element counts match
        from collections import Counter
        assert Counter(result) == Counter(data)

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_minimum_element_first(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test that minimum element is always first after sorting."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = random_data(100, seed=42)

        result = algorithm.execute(data, collector)

        assert result[0] == min(data)

    @pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
    def test_maximum_element_last(
        self, algorithm_class: Type[Algorithm], name: str
    ) -> None:
        """Test that maximum element is always last after sorting."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = random_data(100, seed=42)

        result = algorithm.execute(data, collector)

        assert result[-1] == max(data)
