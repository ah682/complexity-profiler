"""
Unit tests for statistical analysis functionality.

This module tests the Statistics dataclass and compute_statistics function
to ensure accurate statistical calculations for performance analysis.
"""

import pytest
import numpy as np
from typing import List

from complexity_profiler.analysis.statistics import Statistics, compute_statistics


class TestStatistics:
    """Test suite for Statistics dataclass."""

    def test_initialization(self) -> None:
        """Test that Statistics initializes with provided values."""
        stats = Statistics(
            mean=100.0,
            median=98.0,
            std_dev=10.0,
            min=85.0,
            max=115.0,
            percentile_25=92.0,
            percentile_75=108.0,
            coefficient_of_variation=0.10,
        )

        assert stats.mean == 100.0
        assert stats.median == 98.0
        assert stats.std_dev == 10.0
        assert stats.min == 85.0
        assert stats.max == 115.0
        assert stats.percentile_25 == 92.0
        assert stats.percentile_75 == 108.0
        assert stats.coefficient_of_variation == 0.10

    def test_is_consistent_below_threshold(self) -> None:
        """Test is_consistent returns True when CV is below threshold."""
        stats = Statistics(
            mean=100.0,
            median=100.0,
            std_dev=10.0,
            min=80.0,
            max=120.0,
            percentile_25=90.0,
            percentile_75=110.0,
            coefficient_of_variation=0.10,  # 10% < 15% default threshold
        )

        assert stats.is_consistent() is True

    def test_is_consistent_above_threshold(self) -> None:
        """Test is_consistent returns False when CV is above threshold."""
        stats = Statistics(
            mean=100.0,
            median=100.0,
            std_dev=20.0,
            min=60.0,
            max=140.0,
            percentile_25=85.0,
            percentile_75=115.0,
            coefficient_of_variation=0.20,  # 20% > 15% default threshold
        )

        assert stats.is_consistent() is False

    def test_is_consistent_custom_threshold(self) -> None:
        """Test is_consistent with custom threshold."""
        stats = Statistics(
            mean=100.0,
            median=100.0,
            std_dev=10.0,
            min=80.0,
            max=120.0,
            percentile_25=90.0,
            percentile_75=110.0,
            coefficient_of_variation=0.10,
        )

        # Should pass with high threshold
        assert stats.is_consistent(threshold=0.20) is True

        # Should fail with low threshold
        assert stats.is_consistent(threshold=0.05) is False

    def test_is_consistent_at_threshold(self) -> None:
        """Test is_consistent when CV equals threshold."""
        stats = Statistics(
            mean=100.0,
            median=100.0,
            std_dev=15.0,
            min=70.0,
            max=130.0,
            percentile_25=87.5,
            percentile_75=112.5,
            coefficient_of_variation=0.15,  # Exactly at threshold
        )

        # Should be False (< not <=)
        assert stats.is_consistent(threshold=0.15) is False

    def test_frozen_dataclass(self) -> None:
        """Test that Statistics is immutable (frozen)."""
        stats = Statistics(
            mean=100.0,
            median=100.0,
            std_dev=10.0,
            min=80.0,
            max=120.0,
            percentile_25=90.0,
            percentile_75=110.0,
            coefficient_of_variation=0.10,
        )

        # Should raise error when trying to modify
        with pytest.raises(AttributeError):
            stats.mean = 200.0  # type: ignore


class TestComputeStatistics:
    """Test suite for compute_statistics function."""

    def test_compute_with_simple_data(self) -> None:
        """Test computing statistics with simple uniform data."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        stats = compute_statistics(values)

        assert stats.mean == 30.0
        assert stats.median == 30.0
        assert stats.min == 10.0
        assert stats.max == 50.0

    def test_compute_with_single_value(self) -> None:
        """Test computing statistics with a single value."""
        values = [42.0]

        stats = compute_statistics(values)

        assert stats.mean == 42.0
        assert stats.median == 42.0
        assert stats.std_dev == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.percentile_25 == 42.0
        assert stats.percentile_75 == 42.0
        assert stats.coefficient_of_variation == 0.0

    def test_compute_with_two_values(self) -> None:
        """Test computing statistics with two values."""
        values = [10.0, 20.0]

        stats = compute_statistics(values)

        assert stats.mean == 15.0
        assert stats.median == 15.0
        assert stats.min == 10.0
        assert stats.max == 20.0
        assert stats.std_dev > 0

    def test_compute_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute statistics on empty list"):
            compute_statistics([])

    def test_compute_with_identical_values(self) -> None:
        """Test computing statistics when all values are identical."""
        values = [42.0, 42.0, 42.0, 42.0, 42.0]

        stats = compute_statistics(values)

        assert stats.mean == 42.0
        assert stats.median == 42.0
        assert stats.std_dev == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.coefficient_of_variation == 0.0

    def test_compute_with_negative_values(self) -> None:
        """Test computing statistics with negative values."""
        values = [-10.0, -5.0, 0.0, 5.0, 10.0]

        stats = compute_statistics(values)

        assert stats.mean == 0.0
        assert stats.median == 0.0
        assert stats.min == -10.0
        assert stats.max == 10.0

    def test_compute_percentiles(self) -> None:
        """Test that percentiles are computed correctly."""
        # Use 100 values for clear percentile boundaries
        values = list(range(1, 101))  # 1 to 100

        stats = compute_statistics([float(x) for x in values])

        # 25th percentile should be around 25.5
        assert 24.0 <= stats.percentile_25 <= 26.0

        # 75th percentile should be around 75.5
        assert 74.0 <= stats.percentile_75 <= 76.0

    def test_compute_coefficient_of_variation(self) -> None:
        """Test that coefficient of variation is computed correctly."""
        values = [100.0, 110.0, 90.0, 105.0, 95.0]

        stats = compute_statistics(values)

        # CV = std_dev / mean
        expected_cv = stats.std_dev / stats.mean
        assert abs(stats.coefficient_of_variation - expected_cv) < 1e-10

    def test_compute_cv_with_zero_mean(self) -> None:
        """Test coefficient of variation when mean is zero."""
        values = [-10.0, -5.0, 0.0, 5.0, 10.0]

        stats = compute_statistics(values)

        # Mean is 0, so CV should be inf if std_dev > 0, or 0 if std_dev == 0
        assert stats.mean == 0.0
        if stats.std_dev > 0:
            assert stats.coefficient_of_variation == float('inf')
        else:
            assert stats.coefficient_of_variation == 0.0

    def test_compute_cv_with_near_zero_mean(self) -> None:
        """Test coefficient of variation with near-zero mean."""
        values = [1e-11, 2e-11, 3e-11]  # Very small positive values

        stats = compute_statistics(values)

        # Should handle near-zero mean gracefully
        assert stats.coefficient_of_variation >= 0

    def test_compute_with_large_values(self) -> None:
        """Test computing statistics with large values."""
        values = [1e10, 1.1e10, 0.9e10, 1.05e10, 0.95e10]

        stats = compute_statistics(values)

        assert stats.mean > 0
        assert stats.min < stats.max
        assert stats.coefficient_of_variation >= 0

    def test_compute_with_known_data(self) -> None:
        """Test computing statistics with known expected values."""
        # Data: [1, 2, 3, 4, 5]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = compute_statistics(values)

        # Known values
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0

        # Sample standard deviation of [1,2,3,4,5] is sqrt(2.5) ≈ 1.5811
        assert abs(stats.std_dev - np.sqrt(2.5)) < 0.001

        # CV = std_dev / mean
        expected_cv = stats.std_dev / 3.0
        assert abs(stats.coefficient_of_variation - expected_cv) < 1e-10

    def test_compute_with_numpy_array(self) -> None:
        """Test that function works with numpy arrays."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        stats = compute_statistics(values.tolist())

        assert stats.mean == 30.0
        assert stats.median == 30.0

    @pytest.mark.parametrize(
        "values,expected_mean",
        [
            ([1.0, 2.0, 3.0], 2.0),
            ([10.0, 20.0, 30.0, 40.0], 25.0),
            ([5.0, 5.0, 5.0, 5.0], 5.0),
            ([0.0, 100.0], 50.0),
        ],
    )
    def test_compute_mean_parametrized(
        self, values: List[float], expected_mean: float
    ) -> None:
        """Test mean computation with various datasets."""
        stats = compute_statistics(values)

        assert abs(stats.mean - expected_mean) < 1e-10

    def test_compute_with_outliers(self) -> None:
        """Test computing statistics with outliers."""
        # Most values around 100, but one outlier at 1000
        values = [95.0, 98.0, 100.0, 102.0, 105.0, 1000.0]

        stats = compute_statistics(values)

        # Mean should be pulled up by outlier
        assert stats.mean > stats.median

        # Max should be the outlier
        assert stats.max == 1000.0

        # Standard deviation should be large due to outlier
        assert stats.std_dev > 100.0

    def test_compute_odd_number_of_values(self) -> None:
        """Test median computation with odd number of values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        stats = compute_statistics(values)

        # Middle value
        assert stats.median == 3.0

    def test_compute_even_number_of_values(self) -> None:
        """Test median computation with even number of values."""
        values = [1.0, 2.0, 3.0, 4.0]

        stats = compute_statistics(values)

        # Average of two middle values
        assert stats.median == 2.5

    def test_compute_preserves_input(self) -> None:
        """Test that compute_statistics doesn't modify input list."""
        values = [5.0, 2.0, 8.0, 1.0, 9.0]
        original = values.copy()

        compute_statistics(values)

        # Input should be unchanged
        assert values == original

    def test_compute_with_floats_and_ints_mixed(self) -> None:
        """Test computing statistics with mixed int and float values."""
        values = [1.0, 2, 3.0, 4, 5.0]  # type: ignore

        stats = compute_statistics(values)  # type: ignore

        assert stats.mean == 3.0
        assert stats.median == 3.0


class TestStatisticsIntegration:
    """Integration tests for statistics with real performance data."""

    def test_statistics_from_algorithm_metrics(self) -> None:
        """Test computing statistics from multiple algorithm runs."""
        from complexity_profiler.algorithms.sorting import InsertionSort
        from complexity_profiler.analysis.metrics import DefaultMetricsCollector

        # Run algorithm multiple times and collect execution times
        execution_times = []
        algorithm = InsertionSort()

        for _ in range(10):
            collector = DefaultMetricsCollector()
            data = list(range(50, 0, -1))

            with collector.timing():
                algorithm.execute(data, collector)

            execution_times.append(collector.get_metrics().execution_time)

        # Compute statistics
        stats = compute_statistics(execution_times)

        assert stats.mean > 0
        assert stats.std_dev >= 0
        assert stats.min <= stats.median <= stats.max

    def test_consistency_check_on_stable_algorithm(self) -> None:
        """Test that consistent algorithm runs produce low CV."""
        from complexity_profiler.algorithms.sorting import MergeSort
        from complexity_profiler.analysis.metrics import DefaultMetricsCollector

        # Run algorithm multiple times on same input
        comparison_counts = []
        algorithm = MergeSort()
        data = list(range(100, 0, -1))

        for _ in range(10):
            collector = DefaultMetricsCollector()
            algorithm.execute(data, collector)
            comparison_counts.append(float(collector.get_metrics().comparisons))

        stats = compute_statistics(comparison_counts)

        # MergeSort should have very consistent comparison counts
        # (exactly the same for same input)
        assert stats.is_consistent(threshold=0.01)  # Very low threshold
        assert stats.std_dev == 0.0  # Should be exactly 0

    def test_statistics_with_performance_data(
        self, performance_data_generator: any
    ) -> None:
        """Test statistics computation with synthetic performance data."""
        sizes, times = performance_data_generator("linear", 20)

        stats = compute_statistics(times.tolist())

        assert stats.mean > 0
        assert stats.min > 0
        assert stats.max > stats.min
        assert stats.coefficient_of_variation >= 0

    @pytest.mark.parametrize("complexity", ["linear", "quadratic", "linearithmic"])
    def test_statistics_across_complexities(
        self, complexity: str, performance_data_generator: any
    ) -> None:
        """Test that statistics work correctly for different complexities."""
        sizes, times = performance_data_generator(complexity, 15)

        stats = compute_statistics(times.tolist())

        # All complexities should produce valid statistics
        assert stats.mean > 0
        assert stats.std_dev >= 0
        assert 0 <= stats.coefficient_of_variation
        assert stats.min <= stats.median <= stats.max
