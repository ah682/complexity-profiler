"""
Unit tests for sorting algorithm implementations.

This module provides comprehensive tests for all sorting algorithms,
verifying correctness, edge case handling, and metrics collection.
"""

from typing import Any
import pytest
from hypothesis import given, strategies as st

from complexity_profiler.algorithms.sorting import (
    MergeSort,
    QuickSort,
    SelectionSort,
    BubbleSort,
    InsertionSort,
    HeapSort,
)
from complexity_profiler.analysis.metrics import DefaultMetricsCollector
from complexity_profiler.algorithms.base import ComplexityClass


class TestMergeSort:
    """Test suite for MergeSort algorithm."""

    @pytest.fixture
    def algorithm(self) -> MergeSort[int]:
        """Provide MergeSort instance."""
        return MergeSort()

    def test_sorts_correctly(
        self, algorithm: MergeSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that MergeSort correctly sorts random data."""
        collector = DefaultMetricsCollector()
        data = sample_data["random"]
        original = data.copy()

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert sorted(original) == result
        assert data == original  # Original data unchanged

    def test_handles_empty_array(self, algorithm: MergeSort[int]) -> None:
        """Test that MergeSort handles empty arrays correctly."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([], collector)

        assert result == []
        assert collector.get_metrics().comparisons == 0

    def test_handles_single_element(self, algorithm: MergeSort[int]) -> None:
        """Test that MergeSort handles single-element arrays."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([42], collector)

        assert result == [42]
        assert collector.get_metrics().comparisons == 0

    def test_handles_duplicates(
        self, algorithm: MergeSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that MergeSort correctly handles duplicate values."""
        collector = DefaultMetricsCollector()
        data = sample_data["duplicates"]

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == len(data)

    def test_metrics_collected(
        self, algorithm: MergeSort[int], sample_data: dict[str, list[int]]
    ) -> None:
        """Test that MergeSort collects metrics during execution."""
        collector = DefaultMetricsCollector()
        data = sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0
        assert metrics.recursive_calls > 0
        assert metrics.accesses > 0

    def test_metadata(self, algorithm: MergeSort[int]) -> None:
        """Test that MergeSort has correct metadata."""
        metadata = algorithm.metadata

        assert metadata.name == "Merge Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.LINEARITHMIC
        assert metadata.stable is True
        assert metadata.in_place is False

    @pytest.mark.parametrize("size", [10, 50, 100, 200])
    def test_different_sizes(self, algorithm: MergeSort[int], size: int, assert_sorted: Any) -> None:
        """Test MergeSort with different input sizes."""
        import random

        collector = DefaultMetricsCollector()
        data = list(range(size))
        random.shuffle(data)

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == size

    def test_already_sorted(
        self, algorithm: MergeSort[int], sample_data: dict[str, list[int]]
    ) -> None:
        """Test MergeSort on already sorted data."""
        collector = DefaultMetricsCollector()
        data = sample_data["sorted"]

        result = algorithm.execute(data, collector)

        assert result == data

    def test_reverse_sorted(
        self, algorithm: MergeSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test MergeSort on reverse sorted data."""
        collector = DefaultMetricsCollector()
        data = sample_data["reverse"]

        result = algorithm.execute(data, collector)

        assert_sorted(result)

    @given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=100))
    @pytest.mark.property
    def test_property_always_sorts(self, algorithm: MergeSort[int], data: list[int]) -> None:
        """Property test: MergeSort always produces sorted output."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute(data, collector)

        assert result == sorted(data)


class TestQuickSort:
    """Test suite for QuickSort algorithm."""

    @pytest.fixture
    def algorithm(self) -> QuickSort[int]:
        """Provide QuickSort instance."""
        return QuickSort()

    def test_sorts_correctly(
        self, algorithm: QuickSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that QuickSort correctly sorts random data."""
        collector = DefaultMetricsCollector()
        data = sample_data["random"]
        original = data.copy()

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert sorted(original) == result

    def test_handles_empty_array(self, algorithm: QuickSort[int]) -> None:
        """Test that QuickSort handles empty arrays correctly."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([], collector)

        assert result == []

    def test_handles_single_element(self, algorithm: QuickSort[int]) -> None:
        """Test that QuickSort handles single-element arrays."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([42], collector)

        assert result == [42]

    def test_handles_duplicates(
        self, algorithm: QuickSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that QuickSort correctly handles duplicate values."""
        collector = DefaultMetricsCollector()
        data = sample_data["duplicates"]

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == len(data)

    def test_metrics_collected(
        self, algorithm: QuickSort[int], sample_data: dict[str, list[int]]
    ) -> None:
        """Test that QuickSort collects metrics during execution."""
        collector = DefaultMetricsCollector()
        data = sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0
        assert metrics.recursive_calls > 0

    def test_metadata(self, algorithm: QuickSort[int]) -> None:
        """Test that QuickSort has correct metadata."""
        metadata = algorithm.metadata

        assert metadata.name == "Quick Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.LINEARITHMIC
        assert metadata.stable is False

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_different_sizes(self, algorithm: QuickSort[int], size: int, assert_sorted: Any) -> None:
        """Test QuickSort with different input sizes."""
        import random

        collector = DefaultMetricsCollector()
        data = list(range(size))
        random.shuffle(data)

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == size

    @given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=50))
    @pytest.mark.property
    def test_property_always_sorts(self, algorithm: QuickSort[int], data: list[int]) -> None:
        """Property test: QuickSort always produces sorted output."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute(data, collector)

        assert result == sorted(data)


class TestSelectionSort:
    """Test suite for SelectionSort algorithm."""

    @pytest.fixture
    def algorithm(self) -> SelectionSort[int]:
        """Provide SelectionSort instance."""
        return SelectionSort()

    def test_sorts_correctly(
        self, algorithm: SelectionSort[int], small_sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that SelectionSort correctly sorts random data."""
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]
        original = data.copy()

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert sorted(original) == result

    def test_handles_empty_array(self, algorithm: SelectionSort[int]) -> None:
        """Test that SelectionSort handles empty arrays correctly."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([], collector)

        assert result == []

    def test_handles_single_element(self, algorithm: SelectionSort[int]) -> None:
        """Test that SelectionSort handles single-element arrays."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([42], collector)

        assert result == [42]

    def test_handles_duplicates(self, algorithm: SelectionSort[int], assert_sorted: Any) -> None:
        """Test that SelectionSort correctly handles duplicate values."""
        collector = DefaultMetricsCollector()
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == len(data)

    def test_metrics_collected(
        self, algorithm: SelectionSort[int], small_sample_data: dict[str, list[int]]
    ) -> None:
        """Test that SelectionSort collects metrics during execution."""
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps >= 0  # May be 0 for sorted data
        assert metrics.accesses > 0

    def test_metadata(self, algorithm: SelectionSort[int]) -> None:
        """Test that SelectionSort has correct metadata."""
        metadata = algorithm.metadata

        assert metadata.name == "Selection Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.QUADRATIC
        assert metadata.stable is False
        assert metadata.in_place is True

    @pytest.mark.parametrize("size", [5, 10, 20])
    def test_different_sizes(
        self, algorithm: SelectionSort[int], size: int, assert_sorted: Any
    ) -> None:
        """Test SelectionSort with different input sizes."""
        import random

        collector = DefaultMetricsCollector()
        data = list(range(size))
        random.shuffle(data)

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == size


class TestBubbleSort:
    """Test suite for BubbleSort algorithm."""

    @pytest.fixture
    def algorithm(self) -> BubbleSort[int]:
        """Provide BubbleSort instance."""
        return BubbleSort()

    def test_sorts_correctly(
        self, algorithm: BubbleSort[int], small_sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that BubbleSort correctly sorts random data."""
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]
        original = data.copy()

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert sorted(original) == result

    def test_handles_empty_array(self, algorithm: BubbleSort[int]) -> None:
        """Test that BubbleSort handles empty arrays correctly."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([], collector)

        assert result == []

    def test_handles_single_element(self, algorithm: BubbleSort[int]) -> None:
        """Test that BubbleSort handles single-element arrays."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([42], collector)

        assert result == [42]

    def test_handles_duplicates(self, algorithm: BubbleSort[int], assert_sorted: Any) -> None:
        """Test that BubbleSort correctly handles duplicate values."""
        collector = DefaultMetricsCollector()
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

        result = algorithm.execute(data, collector)

        assert_sorted(result)

    def test_metrics_collected(
        self, algorithm: BubbleSort[int], small_sample_data: dict[str, list[int]]
    ) -> None:
        """Test that BubbleSort collects metrics during execution."""
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0

    def test_metadata(self, algorithm: BubbleSort[int]) -> None:
        """Test that BubbleSort has correct metadata."""
        metadata = algorithm.metadata

        assert metadata.name == "Bubble Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.QUADRATIC
        assert metadata.stable is True
        assert metadata.in_place is True

    def test_early_termination_optimization(self, algorithm: BubbleSort[int]) -> None:
        """Test that BubbleSort terminates early on sorted data."""
        collector = DefaultMetricsCollector()
        data = [1, 2, 3, 4, 5]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        # Should terminate after first pass with no swaps
        assert metrics.comparisons < len(data) * (len(data) - 1) // 2


class TestInsertionSort:
    """Test suite for InsertionSort algorithm."""

    @pytest.fixture
    def algorithm(self) -> InsertionSort[int]:
        """Provide InsertionSort instance."""
        return InsertionSort()

    def test_sorts_correctly(
        self, algorithm: InsertionSort[int], small_sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that InsertionSort correctly sorts random data."""
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]
        original = data.copy()

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert sorted(original) == result

    def test_handles_empty_array(self, algorithm: InsertionSort[int]) -> None:
        """Test that InsertionSort handles empty arrays correctly."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([], collector)

        assert result == []

    def test_handles_single_element(self, algorithm: InsertionSort[int]) -> None:
        """Test that InsertionSort handles single-element arrays."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([42], collector)

        assert result == [42]

    def test_handles_duplicates(self, algorithm: InsertionSort[int], assert_sorted: Any) -> None:
        """Test that InsertionSort correctly handles duplicate values."""
        collector = DefaultMetricsCollector()
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

        result = algorithm.execute(data, collector)

        assert_sorted(result)

    def test_metrics_collected(
        self, algorithm: InsertionSort[int], small_sample_data: dict[str, list[int]]
    ) -> None:
        """Test that InsertionSort collects metrics during execution."""
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0

    def test_metadata(self, algorithm: InsertionSort[int]) -> None:
        """Test that InsertionSort has correct metadata."""
        metadata = algorithm.metadata

        assert metadata.name == "Insertion Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.QUADRATIC
        assert metadata.stable is True
        assert metadata.in_place is True

    def test_efficient_on_nearly_sorted(self, algorithm: InsertionSort[int]) -> None:
        """Test that InsertionSort is efficient on nearly sorted data."""
        collector = DefaultMetricsCollector()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]  # Only last two out of order

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        # Should have fewer comparisons than worst case
        assert metrics.comparisons < len(data) * (len(data) - 1) // 2


class TestHeapSort:
    """Test suite for HeapSort algorithm."""

    @pytest.fixture
    def algorithm(self) -> HeapSort[int]:
        """Provide HeapSort instance."""
        return HeapSort()

    def test_sorts_correctly(
        self, algorithm: HeapSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that HeapSort correctly sorts random data."""
        collector = DefaultMetricsCollector()
        data = sample_data["random"]
        original = data.copy()

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert sorted(original) == result

    def test_handles_empty_array(self, algorithm: HeapSort[int]) -> None:
        """Test that HeapSort handles empty arrays correctly."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([], collector)

        assert result == []

    def test_handles_single_element(self, algorithm: HeapSort[int]) -> None:
        """Test that HeapSort handles single-element arrays."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute([42], collector)

        assert result == [42]

    def test_handles_duplicates(
        self, algorithm: HeapSort[int], sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that HeapSort correctly handles duplicate values."""
        collector = DefaultMetricsCollector()
        data = sample_data["duplicates"]

        result = algorithm.execute(data, collector)

        assert_sorted(result)

    def test_metrics_collected(
        self, algorithm: HeapSort[int], sample_data: dict[str, list[int]]
    ) -> None:
        """Test that HeapSort collects metrics during execution."""
        collector = DefaultMetricsCollector()
        data = sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        assert metrics.comparisons > 0
        assert metrics.swaps > 0
        assert metrics.recursive_calls >= 0  # May use recursion in heapify

    def test_metadata(self, algorithm: HeapSort[int]) -> None:
        """Test that HeapSort has correct metadata."""
        metadata = algorithm.metadata

        assert metadata.name == "Heap Sort"
        assert metadata.category == "sorting"
        assert metadata.expected_complexity == ComplexityClass.LINEARITHMIC
        assert metadata.stable is False
        assert metadata.in_place is True

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_different_sizes(self, algorithm: HeapSort[int], size: int, assert_sorted: Any) -> None:
        """Test HeapSort with different input sizes."""
        import random

        collector = DefaultMetricsCollector()
        data = list(range(size))
        random.shuffle(data)

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == size

    @given(st.lists(st.integers(min_value=-1000, max_value=1000), max_size=100))
    @pytest.mark.property
    def test_property_always_sorts(self, algorithm: HeapSort[int], data: list[int]) -> None:
        """Property test: HeapSort always produces sorted output."""
        collector = DefaultMetricsCollector()
        result = algorithm.execute(data, collector)

        assert result == sorted(data)


class TestAllAlgorithms:
    """Cross-algorithm tests to verify all algorithms behave consistently."""

    @pytest.mark.parametrize(
        "algorithm_class",
        [MergeSort, QuickSort, SelectionSort, BubbleSort, InsertionSort, HeapSort],
    )
    def test_all_algorithms_sort_correctly(
        self, algorithm_class: Any, sample_data: dict[str, list[int]], assert_sorted: Any
    ) -> None:
        """Test that all algorithms produce correctly sorted output."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = sample_data["random"][:50]  # Use smaller dataset for slow algorithms

        result = algorithm.execute(data, collector)

        assert_sorted(result)
        assert len(result) == len(data)

    @pytest.mark.parametrize(
        "algorithm_class",
        [MergeSort, QuickSort, SelectionSort, BubbleSort, InsertionSort, HeapSort],
    )
    def test_all_algorithms_have_metadata(self, algorithm_class: Any) -> None:
        """Test that all algorithms provide valid metadata."""
        algorithm = algorithm_class()
        metadata = algorithm.metadata

        assert metadata.name
        assert metadata.category == "sorting"
        assert isinstance(metadata.expected_complexity, ComplexityClass)
        assert metadata.space_complexity

    @pytest.mark.parametrize(
        "algorithm_class",
        [MergeSort, QuickSort, SelectionSort, BubbleSort, InsertionSort, HeapSort],
    )
    def test_all_algorithms_collect_metrics(
        self, algorithm_class: Any, small_sample_data: dict[str, list[int]]
    ) -> None:
        """Test that all algorithms collect metrics during execution."""
        algorithm = algorithm_class()
        collector = DefaultMetricsCollector()
        data = small_sample_data["random"]

        algorithm.execute(data, collector)
        metrics = collector.get_metrics()

        # All algorithms should perform at least some operations
        assert metrics.total_operations() > 0
