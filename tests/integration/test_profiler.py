"""
Integration tests for the algorithm profiling workflow.

This module tests the complete profiling pipeline including:
- Algorithm profiling with multiple input sizes
- Statistical analysis of results
- Complexity curve fitting
- End-to-end profiling workflows
- ProfileResult serialization and reporting
"""

import pytest
import numpy as np
from typing import Callable, Any

from complexity_profiler.analysis.profiler import (
    AlgorithmProfiler,
    ProfileResult,
    profile_algorithm,
)
from complexity_profiler.algorithms.sorting import (
    MergeSort,
    QuickSort,
    SelectionSort,
    BubbleSort,
    InsertionSort,
    HeapSort,
)
from complexity_profiler.algorithms.base import ComplexityClass
from complexity_profiler.data.generators import (
    random_data,
    sorted_data,
    reverse_sorted_data,
)


class TestAlgorithmProfiler:
    """Test suite for AlgorithmProfiler class."""

    def test_profiler_initialization(self) -> None:
        """Test that profiler initializes with correct default values."""
        profiler = AlgorithmProfiler()

        assert profiler._input_sizes == [10, 50, 100, 500, 1000]
        assert profiler._runs_per_size == 3
        assert profiler._data_generator is None
        assert profiler._warmup_runs == 2
        assert profiler._min_r_squared == 0.8

    def test_profiler_with_custom_sizes(self) -> None:
        """Test setting custom input sizes."""
        profiler = AlgorithmProfiler()
        sizes = [20, 50, 100, 200]

        result = profiler.with_sizes(sizes)

        assert result is profiler  # Check fluent API
        assert profiler._input_sizes == sizes

    def test_profiler_with_custom_runs(self) -> None:
        """Test setting custom number of runs."""
        profiler = AlgorithmProfiler()

        result = profiler.with_runs(5)

        assert result is profiler
        assert profiler._runs_per_size == 5

    def test_profiler_with_data_generator(self) -> None:
        """Test setting custom data generator."""
        profiler = AlgorithmProfiler()
        generator = lambda n: list(range(n))

        result = profiler.with_data_generator(generator)

        assert result is profiler
        assert profiler._data_generator is not None

    def test_profiler_sizes_validation_empty(self) -> None:
        """Test that empty sizes list raises ValueError."""
        profiler = AlgorithmProfiler()

        with pytest.raises(ValueError, match="sizes list cannot be empty"):
            profiler.with_sizes([])

    def test_profiler_sizes_validation_non_positive(self) -> None:
        """Test that non-positive sizes raise ValueError."""
        profiler = AlgorithmProfiler()

        with pytest.raises(ValueError, match="all sizes must be positive"):
            profiler.with_sizes([10, 20, 0, 30])

        with pytest.raises(ValueError, match="all sizes must be positive"):
            profiler.with_sizes([10, -5, 20])

    def test_profiler_runs_validation(self) -> None:
        """Test that non-positive runs raise ValueError."""
        profiler = AlgorithmProfiler()

        with pytest.raises(ValueError, match="runs must be positive"):
            profiler.with_runs(0)

        with pytest.raises(ValueError, match="runs must be positive"):
            profiler.with_runs(-1)

    def test_profile_without_data_generator_raises_error(self) -> None:
        """Test that profiling without data generator raises ValueError."""
        profiler = AlgorithmProfiler()
        algorithm = MergeSort()

        with pytest.raises(ValueError, match="Data generator must be set"):
            profiler.profile(algorithm)

    def test_profile_merge_sort_detects_linearithmic(self) -> None:
        """Test profiling MergeSort detects O(n log n) complexity."""
        profiler = AlgorithmProfiler()
        algorithm = MergeSort()

        result = (profiler
                  .with_sizes([50, 100, 200, 400, 800])
                  .with_runs(3)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert isinstance(result, ProfileResult)
        assert result.algorithm_name == "Merge Sort"
        assert result.category == "sorting"
        assert result.empirical_complexity == str(ComplexityClass.LINEARITHMIC)
        assert result.r_squared >= 0.8

    def test_profile_selection_sort_detects_quadratic(self) -> None:
        """Test profiling SelectionSort detects O(n²) complexity."""
        profiler = AlgorithmProfiler()
        algorithm = SelectionSort()

        result = (profiler
                  .with_sizes([20, 40, 60, 80, 100])
                  .with_runs(3)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert result.algorithm_name == "Selection Sort"
        assert result.empirical_complexity == str(ComplexityClass.QUADRATIC)
        assert result.r_squared >= 0.8

    def test_profile_collects_metrics_for_all_sizes(self) -> None:
        """Test that profiling collects metrics for all input sizes."""
        profiler = AlgorithmProfiler()
        algorithm = InsertionSort()
        sizes = [10, 20, 30, 40, 50]

        result = (profiler
                  .with_sizes(sizes)
                  .with_runs(3)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert len(result.metrics_per_size) == len(sizes)
        assert all(size in result.metrics_per_size for size in sizes)

        # Each size should have correct number of runs
        for size in sizes:
            assert len(result.metrics_per_size[size]) == 3

    def test_profile_collects_statistics_for_all_sizes(self) -> None:
        """Test that profiling collects statistics for all input sizes."""
        profiler = AlgorithmProfiler()
        algorithm = BubbleSort()
        sizes = [10, 20, 30, 40]

        result = (profiler
                  .with_sizes(sizes)
                  .with_runs(5)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert len(result.statistics_per_size) == len(sizes)
        assert all(size in result.statistics_per_size for size in sizes)

        # Each size should have valid statistics
        for size in sizes:
            stats = result.statistics_per_size[size]
            assert stats.mean > 0
            assert stats.std_dev >= 0
            assert stats.min <= stats.median <= stats.max

    def test_profile_with_different_data_generators(self) -> None:
        """Test profiling with different data generator types."""
        algorithm = QuickSort()

        # Random data
        profiler_random = AlgorithmProfiler()
        result_random = (profiler_random
                         .with_sizes([50, 100, 200])
                         .with_runs(3)
                         .with_data_generator(lambda n: random_data(n, seed=42))
                         .profile(algorithm))

        # Sorted data (potentially worst case for some quicksort implementations)
        profiler_sorted = AlgorithmProfiler()
        result_sorted = (profiler_sorted
                         .with_sizes([50, 100, 200])
                         .with_runs(3)
                         .with_data_generator(lambda n: sorted_data(n))
                         .profile(algorithm))

        # Both should complete successfully
        assert result_random.empirical_complexity is not None
        assert result_sorted.empirical_complexity is not None

    def test_profile_multiple_runs_improves_consistency(self) -> None:
        """Test that multiple runs produce consistent statistics."""
        profiler = AlgorithmProfiler()
        algorithm = MergeSort()

        result = (profiler
                  .with_sizes([100, 200, 300])
                  .with_runs(10)  # More runs
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        # With more runs, coefficient of variation should be low
        for size in result.input_sizes:
            stats = result.statistics_per_size[size]
            # MergeSort should have very consistent comparison counts
            assert stats.coefficient_of_variation < 0.2


class TestProfileResult:
    """Test suite for ProfileResult dataclass."""

    @pytest.fixture
    def sample_profile_result(self) -> ProfileResult:
        """Create a sample ProfileResult for testing."""
        from complexity_profiler.analysis.metrics import PerformanceMetrics
        from complexity_profiler.analysis.statistics import Statistics

        sizes = [10, 20, 30]
        metrics_per_size = {
            10: [PerformanceMetrics(comparisons=50, swaps=25, execution_time=0.001)],
            20: [PerformanceMetrics(comparisons=200, swaps=100, execution_time=0.004)],
            30: [PerformanceMetrics(comparisons=450, swaps=225, execution_time=0.009)],
        }
        statistics_per_size = {
            10: Statistics(mean=0.001, median=0.001, std_dev=0.0001,
                          min=0.001, max=0.001, percentile_25=0.001,
                          percentile_75=0.001, coefficient_of_variation=0.1),
            20: Statistics(mean=0.004, median=0.004, std_dev=0.0004,
                          min=0.004, max=0.004, percentile_25=0.004,
                          percentile_75=0.004, coefficient_of_variation=0.1),
            30: Statistics(mean=0.009, median=0.009, std_dev=0.0009,
                          min=0.009, max=0.009, percentile_25=0.009,
                          percentile_75=0.009, coefficient_of_variation=0.1),
        }

        return ProfileResult(
            algorithm_name="Test Algorithm",
            category="sorting",
            input_sizes=sizes,
            metrics_per_size=metrics_per_size,
            statistics_per_size=statistics_per_size,
            empirical_complexity="O(n²)",
            r_squared=0.95,
            notes="Test profile"
        )

    def test_profile_result_initialization(
        self, sample_profile_result: ProfileResult
    ) -> None:
        """Test ProfileResult initialization."""
        assert sample_profile_result.algorithm_name == "Test Algorithm"
        assert sample_profile_result.category == "sorting"
        assert sample_profile_result.empirical_complexity == "O(n²)"
        assert sample_profile_result.r_squared == 0.95
        assert sample_profile_result.notes == "Test profile"

    def test_profile_result_validation_empty_sizes(self) -> None:
        """Test that empty input_sizes raises ValueError."""
        from complexity_profiler.analysis.metrics import PerformanceMetrics
        from complexity_profiler.analysis.statistics import Statistics

        with pytest.raises(ValueError, match="input_sizes cannot be empty"):
            ProfileResult(
                algorithm_name="Test",
                category="sorting",
                input_sizes=[],
                metrics_per_size={},
                statistics_per_size={},
                empirical_complexity="O(n)",
                r_squared=0.9
            )

    def test_profile_result_validation_empty_name(self) -> None:
        """Test that empty algorithm_name raises ValueError."""
        from complexity_profiler.analysis.metrics import PerformanceMetrics
        from complexity_profiler.analysis.statistics import Statistics

        with pytest.raises(ValueError, match="algorithm_name cannot be empty"):
            ProfileResult(
                algorithm_name="",
                category="sorting",
                input_sizes=[10, 20],
                metrics_per_size={},
                statistics_per_size={},
                empirical_complexity="O(n)",
                r_squared=0.9
            )

    def test_profile_result_validation_invalid_r_squared(self) -> None:
        """Test that invalid r_squared raises ValueError."""
        from complexity_profiler.analysis.metrics import PerformanceMetrics
        from complexity_profiler.analysis.statistics import Statistics

        with pytest.raises(ValueError, match="r_squared must be between 0 and 1"):
            ProfileResult(
                algorithm_name="Test",
                category="sorting",
                input_sizes=[10, 20],
                metrics_per_size={},
                statistics_per_size={},
                empirical_complexity="O(n)",
                r_squared=1.5
            )

    def test_profile_result_to_dict(
        self, sample_profile_result: ProfileResult
    ) -> None:
        """Test conversion of ProfileResult to dictionary."""
        result_dict = sample_profile_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["algorithm_name"] == "Test Algorithm"
        assert result_dict["category"] == "sorting"
        assert result_dict["empirical_complexity"] == "O(n²)"
        assert result_dict["r_squared"] == 0.95

        # Check timestamp is converted to ISO format
        assert isinstance(result_dict["timestamp"], str)

        # Check metrics are converted to dicts
        assert "10" in result_dict["metrics_per_size"]
        assert isinstance(result_dict["metrics_per_size"]["10"], list)

    def test_profile_result_get_summary(
        self, sample_profile_result: ProfileResult
    ) -> None:
        """Test getting summary from ProfileResult."""
        summary = sample_profile_result.get_summary()

        assert isinstance(summary, dict)
        assert summary["algorithm_name"] == "Test Algorithm"
        assert summary["category"] == "sorting"
        assert summary["input_sizes"] == [10, 20, 30]
        assert summary["empirical_complexity"] == "O(n²)"
        assert summary["r_squared"] == 0.95
        assert "timestamp" in summary


class TestProfileAlgorithmConvenience:
    """Test suite for convenience profile_algorithm function."""

    def test_profile_algorithm_basic(self) -> None:
        """Test basic usage of profile_algorithm convenience function."""
        algorithm = MergeSort()
        sizes = [50, 100, 200]

        result = profile_algorithm(
            algorithm,
            sizes=sizes,
            runs=3,
            data_generator=lambda n: random_data(n, seed=42)
        )

        assert isinstance(result, ProfileResult)
        assert result.algorithm_name == "Merge Sort"
        assert result.input_sizes == sizes
        assert result.r_squared > 0.0

    def test_profile_algorithm_with_default_generator(self) -> None:
        """Test profile_algorithm with default random data generator."""
        algorithm = QuickSort()
        sizes = [20, 40, 60]

        # Should use default random generator
        result = profile_algorithm(algorithm, sizes=sizes, runs=3)

        assert isinstance(result, ProfileResult)
        assert result.algorithm_name == "Quick Sort"
        assert len(result.metrics_per_size) == len(sizes)

    def test_profile_algorithm_custom_runs(self) -> None:
        """Test profile_algorithm with custom number of runs."""
        algorithm = InsertionSort()
        sizes = [20, 40]

        result = profile_algorithm(
            algorithm,
            sizes=sizes,
            runs=7,
            data_generator=lambda n: reverse_sorted_data(n)
        )

        # Should have 7 runs per size
        for size in sizes:
            assert len(result.metrics_per_size[size]) == 7


class TestEndToEndProfiling:
    """End-to-end integration tests for complete profiling workflows."""

    @pytest.mark.parametrize("algorithm_class,expected_complexity", [
        (MergeSort, ComplexityClass.LINEARITHMIC),
        (QuickSort, ComplexityClass.LINEARITHMIC),
        (HeapSort, ComplexityClass.LINEARITHMIC),
        (SelectionSort, ComplexityClass.QUADRATIC),
        (BubbleSort, ComplexityClass.QUADRATIC),
        (InsertionSort, ComplexityClass.QUADRATIC),
    ])
    def test_profile_detects_correct_complexity(
        self,
        algorithm_class: type,
        expected_complexity: ComplexityClass
    ) -> None:
        """Test that profiling correctly identifies algorithm complexity."""
        algorithm = algorithm_class()
        profiler = AlgorithmProfiler()

        # Use appropriate sizes for complexity
        if expected_complexity == ComplexityClass.QUADRATIC:
            sizes = [20, 40, 60, 80, 100]
        else:
            sizes = [100, 200, 400, 800, 1600]

        result = (profiler
                  .with_sizes(sizes)
                  .with_runs(3)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert result.empirical_complexity == str(expected_complexity)
        assert result.r_squared >= 0.75  # Allow slightly lower R² for real data

    def test_profile_with_warmup_runs(self) -> None:
        """Test that warmup runs don't affect results."""
        algorithm = MergeSort()
        profiler = AlgorithmProfiler(warmup_runs=3)

        result = (profiler
                  .with_sizes([100, 200, 300])
                  .with_runs(5)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        # Should have exactly 5 runs per size (warmup not included)
        for size in result.input_sizes:
            assert len(result.metrics_per_size[size]) == 5

    def test_profile_different_input_patterns(self) -> None:
        """Test profiling with different input data patterns."""
        algorithm = BubbleSort()
        profiler = AlgorithmProfiler()
        sizes = [20, 40, 60]

        # Profile with random data
        result_random = (profiler
                         .with_sizes(sizes)
                         .with_runs(3)
                         .with_data_generator(lambda n: random_data(n, seed=42))
                         .profile(algorithm))

        # Profile with sorted data (best case for bubble sort)
        result_sorted = (profiler
                         .with_sizes(sizes)
                         .with_runs(3)
                         .with_data_generator(lambda n: sorted_data(n))
                         .profile(algorithm))

        # Profile with reverse sorted data (worst case)
        result_reverse = (profiler
                          .with_sizes(sizes)
                          .with_runs(3)
                          .with_data_generator(lambda n: reverse_sorted_data(n))
                          .profile(algorithm))

        # All should complete and produce valid results
        assert result_random.r_squared > 0.0
        assert result_sorted.r_squared > 0.0
        assert result_reverse.r_squared > 0.0

        # Sorted should be faster than reverse sorted
        sorted_mean_time = result_sorted.statistics_per_size[sizes[-1]].mean
        reverse_mean_time = result_reverse.statistics_per_size[sizes[-1]].mean
        assert sorted_mean_time < reverse_mean_time

    def test_profile_comprehensive_workflow(self) -> None:
        """Test complete profiling workflow from start to finish."""
        # Setup
        algorithm = HeapSort()
        profiler = AlgorithmProfiler(warmup_runs=2, min_r_squared=0.8)

        # Configure
        sizes = [100, 200, 400, 800]
        profiler.with_sizes(sizes)
        profiler.with_runs(5)
        profiler.with_data_generator(lambda n: random_data(n, seed=42))

        # Profile
        result = profiler.profile(algorithm)

        # Verify results structure
        assert result.algorithm_name == "Heap Sort"
        assert result.category == "sorting"
        assert result.input_sizes == sizes
        assert len(result.metrics_per_size) == len(sizes)
        assert len(result.statistics_per_size) == len(sizes)

        # Verify metrics collected
        for size in sizes:
            metrics_list = result.metrics_per_size[size]
            assert len(metrics_list) == 5
            for metrics in metrics_list:
                assert metrics.comparisons > 0
                assert metrics.execution_time > 0

        # Verify statistics computed
        for size in sizes:
            stats = result.statistics_per_size[size]
            assert stats.mean > 0
            assert stats.std_dev >= 0
            assert stats.min <= stats.mean <= stats.max

        # Verify complexity analysis
        assert result.empirical_complexity == str(ComplexityClass.LINEARITHMIC)
        assert 0.0 <= result.r_squared <= 1.0

        # Verify serialization works
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        summary = result.get_summary()
        assert isinstance(summary, dict)
        assert "algorithm_name" in summary
        assert "empirical_complexity" in summary

    def test_profile_comparison_across_algorithms(self) -> None:
        """Test comparing multiple algorithms through profiling."""
        sizes = [50, 100, 200]
        algorithms = [
            MergeSort(),
            QuickSort(),
            InsertionSort(),
        ]

        results = []
        for algorithm in algorithms:
            profiler = AlgorithmProfiler()
            result = (profiler
                      .with_sizes(sizes)
                      .with_runs(3)
                      .with_data_generator(lambda n: random_data(n, seed=42))
                      .profile(algorithm))
            results.append(result)

        # All should complete successfully
        assert len(results) == 3

        # MergeSort and QuickSort should be O(n log n)
        assert results[0].empirical_complexity == str(ComplexityClass.LINEARITHMIC)
        assert results[1].empirical_complexity == str(ComplexityClass.LINEARITHMIC)

        # InsertionSort should be O(n²)
        assert results[2].empirical_complexity == str(ComplexityClass.QUADRATIC)

    @pytest.mark.slow
    def test_profile_with_large_input_sizes(self) -> None:
        """Test profiling with large input sizes (slow test)."""
        algorithm = MergeSort()
        profiler = AlgorithmProfiler()

        result = (profiler
                  .with_sizes([1000, 2000, 4000, 8000])
                  .with_runs(3)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert result.empirical_complexity == str(ComplexityClass.LINEARITHMIC)
        assert result.r_squared >= 0.8

    def test_profile_stores_timestamp(self) -> None:
        """Test that profile result stores timestamp."""
        algorithm = InsertionSort()
        profiler = AlgorithmProfiler()

        result = (profiler
                  .with_sizes([20, 40])
                  .with_runs(2)
                  .with_data_generator(lambda n: random_data(n, seed=42))
                  .profile(algorithm))

        assert result.timestamp is not None
        # Timestamp should be recent (within last minute)
        from datetime import datetime, timedelta
        assert (datetime.now() - result.timestamp) < timedelta(minutes=1)


class TestProfilingErrorHandling:
    """Test suite for error handling in profiling."""

    def test_profile_with_insufficient_data_points(self) -> None:
        """Test that profiling with too few data points fails gracefully."""
        algorithm = MergeSort()
        profiler = AlgorithmProfiler()

        # Only 2 sizes is not enough for curve fitting (need at least 3)
        with pytest.raises((ValueError, RuntimeError)):
            profiler.with_sizes([10, 20]).with_runs(3).with_data_generator(
                lambda n: random_data(n, seed=42)
            ).profile(algorithm)

    def test_profile_with_invalid_data_generator(self) -> None:
        """Test profiling fails with invalid data generator."""
        algorithm = MergeSort()
        profiler = AlgorithmProfiler()

        # Generator that returns wrong type
        def bad_generator(n: int) -> Any:
            return "not a list"

        with pytest.raises(Exception):  # Will fail in algorithm.execute
            profiler.with_sizes([10, 20, 30]).with_runs(2).with_data_generator(
                bad_generator
            ).profile(algorithm)
