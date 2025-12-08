"""
Statistical analysis utilities for performance metrics.

This module provides tools for computing and analyzing statistical properties
of performance measurements, including measures of central tendency, dispersion,
and consistency checks.
"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Statistics:
    """
    Statistical summary of a dataset.

    Attributes:
        mean: Arithmetic mean of the values
        median: Middle value when sorted
        std_dev: Standard deviation (measure of spread)
        min: Minimum value
        max: Maximum value
        percentile_25: 25th percentile (Q1)
        percentile_75: 75th percentile (Q3)
        coefficient_of_variation: Ratio of std_dev to mean (CV)
    """

    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    percentile_25: float
    percentile_75: float
    coefficient_of_variation: float

    def is_consistent(self, threshold: float = 0.15) -> bool:
        """
        Check if the measurements are consistent based on coefficient of variation.

        A lower CV indicates more consistent measurements. The default threshold
        of 0.15 (15%) is commonly used in experimental analysis.

        Args:
            threshold: Maximum acceptable CV for consistency (default: 0.15)

        Returns:
            True if CV is below threshold, indicating consistent measurements

        Example:
            >>> stats = Statistics(mean=100, median=98, std_dev=10,
            ...                    min=85, max=115, percentile_25=92,
            ...                    percentile_75=108, coefficient_of_variation=0.10)
            >>> stats.is_consistent()
            True
            >>> stats.is_consistent(threshold=0.05)
            False
        """
        return self.coefficient_of_variation < threshold


def compute_statistics(values: list[float]) -> Statistics:
    """
    Compute comprehensive statistics from a list of values.

    Calculates various statistical measures including central tendency,
    spread, and percentiles. Handles edge cases like empty lists and
    zero mean gracefully.

    Args:
        values: List of numeric values to analyze

    Returns:
        Statistics object containing all computed metrics

    Raises:
        ValueError: If values list is empty

    Example:
        >>> values = [10.5, 12.3, 11.8, 10.9, 12.1]
        >>> stats = compute_statistics(values)
        >>> print(f"Mean: {stats.mean:.2f}, Std Dev: {stats.std_dev:.2f}")
        Mean: 11.52, Std Dev: 0.70
    """
    if not values:
        raise ValueError("Cannot compute statistics on empty list")

    # Convert to numpy array for efficient computation
    arr: npt.NDArray[np.float64] = np.array(values, dtype=np.float64)

    # Compute basic statistics
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_dev_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Compute percentiles
    percentile_25_val = float(np.percentile(arr, 25))
    percentile_75_val = float(np.percentile(arr, 75))

    # Compute coefficient of variation
    # Handle division by zero if mean is zero
    if abs(mean_val) < 1e-10:
        cv = 0.0 if std_dev_val < 1e-10 else float('inf')
    else:
        cv = std_dev_val / abs(mean_val)

    return Statistics(
        mean=mean_val,
        median=median_val,
        std_dev=std_dev_val,
        min=min_val,
        max=max_val,
        percentile_25=percentile_25_val,
        percentile_75=percentile_75_val,
        coefficient_of_variation=cv
    )


__all__ = [
    "Statistics",
    "compute_statistics",
]
