"""
Complexity curve fitting for Big-O analysis.

This module provides tools for fitting empirical performance data to theoretical
complexity curves and determining the best-fit complexity class.
"""

from typing import Callable, Dict, Tuple, Optional
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from complexity_profiler.algorithms.base import ComplexityClass


# Type alias for complexity function signature
ComplexityFunction = Callable[[npt.NDArray[np.float64], float, float], npt.NDArray[np.float64]]


class ComplexityFitter:
    """
    Fits empirical performance data to theoretical complexity curves.

    This class attempts to fit the measured execution times to various
    complexity functions and determines which complexity class best
    describes the algorithm's behavior.

    The fitting process uses non-linear least squares optimization and
    evaluates fit quality using R-squared (coefficient of determination).

    Example:
        >>> fitter = ComplexityFitter()
        >>> sizes = np.array([10, 20, 30, 40, 50])
        >>> times = np.array([0.001, 0.004, 0.009, 0.016, 0.025])
        >>> complexity, r_squared = fitter.fit_complexity(sizes, times)
        >>> print(f"Best fit: {complexity} with R�={r_squared:.4f}")
        Best fit: O(n�) with R�=1.0000
    """

    # Dictionary mapping complexity classes to their mathematical functions
    COMPLEXITY_FUNCTIONS: Dict[ComplexityClass, ComplexityFunction] = {
        ComplexityClass.CONSTANT: lambda n, a, b: a * np.ones_like(n) + b,
        ComplexityClass.LOGARITHMIC: lambda n, a, b: a * np.log(n) + b,
        ComplexityClass.LINEAR: lambda n, a, b: a * n + b,
        ComplexityClass.LINEARITHMIC: lambda n, a, b: a * n * np.log(n) + b,
        ComplexityClass.QUADRATIC: lambda n, a, b: a * n**2 + b,
        ComplexityClass.CUBIC: lambda n, a, b: a * n**3 + b,
    }

    def __init__(self, min_r_squared: float = 0.8) -> None:
        """
        Initialize the complexity fitter.

        Args:
            min_r_squared: Minimum R� value to consider a fit acceptable (default: 0.8)
        """
        self.min_r_squared = min_r_squared

    def fit_complexity(
        self,
        sizes: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> Tuple[ComplexityClass, float]:
        """
        Fit performance data to complexity curves and return the best fit.

        Attempts to fit the data to each complexity function and selects
        the one with the highest R� value. Prefers simpler complexities
        when R� values are very close.

        Args:
            sizes: Array of input sizes (n values)
            times: Array of execution times corresponding to each size

        Returns:
            Tuple of (best_complexity_class, r_squared_score)

        Raises:
            ValueError: If sizes and times have different lengths or are too short
            RuntimeError: If no acceptable fit could be found

        Example:
            >>> sizes = np.array([100, 200, 300, 400, 500], dtype=np.float64)
            >>> times = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
            >>> fitter = ComplexityFitter()
            >>> complexity, r_sq = fitter.fit_complexity(sizes, times)
            >>> print(complexity)
            O(n)
        """
        if len(sizes) != len(times):
            raise ValueError("sizes and times must have the same length")

        if len(sizes) < 3:
            raise ValueError("Need at least 3 data points for curve fitting")

        # Ensure arrays are float64
        sizes_arr = np.asarray(sizes, dtype=np.float64)
        times_arr = np.asarray(times, dtype=np.float64)

        # Try to fit each complexity function
        fits: Dict[ComplexityClass, float] = {}

        for complexity_class, func in self.COMPLEXITY_FUNCTIONS.items():
            try:
                r_squared = self._fit_single_complexity(
                    sizes_arr, times_arr, func
                )
                fits[complexity_class] = r_squared
            except (RuntimeError, ValueError, TypeError):
                # Fitting failed for this complexity, skip it
                continue

        if not fits:
            raise RuntimeError("Could not fit data to any complexity function")

        # Find the best fit
        best_complexity = max(fits.items(), key=lambda x: x[1])

        # Apply Occam's Razor: prefer simpler complexity if R� is very close
        best_class, best_r_sq = best_complexity
        best_class = self._apply_simplicity_bias(fits, best_class, best_r_sq)

        return best_class, fits[best_class]

    def _fit_single_complexity(
        self,
        sizes: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        func: ComplexityFunction,
    ) -> float:
        """
        Fit data to a single complexity function and return R� score.

        Args:
            sizes: Array of input sizes
            times: Array of execution times
            func: Complexity function to fit

        Returns:
            R-squared value indicating fit quality

        Raises:
            RuntimeError: If curve fitting fails
            ValueError: If R� calculation fails
        """
        # Initial parameter guesses
        p0 = [1.0, 0.0]

        # Attempt curve fitting with bounds to keep parameters reasonable
        params, _ = curve_fit(
            func,
            sizes,
            times,
            p0=p0,
            maxfev=10000,
            bounds=([-np.inf, -np.inf], [np.inf, np.inf])
        )

        # Calculate predicted values
        predicted = func(sizes, *params)

        # Calculate R-squared
        r_squared = self._calculate_r_squared(times, predicted)

        return r_squared

    def _calculate_r_squared(
        self,
        actual: npt.NDArray[np.float64],
        predicted: npt.NDArray[np.float64],
    ) -> float:
        """
        Calculate the coefficient of determination (R�).

        R� measures how well the predicted values match the actual values.
        Values range from - to 1, where 1 indicates perfect fit.

        Args:
            actual: Actual measured values
            predicted: Predicted values from the fitted curve

        Returns:
            R-squared value
        """
        # Calculate residual sum of squares
        ss_res = np.sum((actual - predicted) ** 2)

        # Calculate total sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        # Handle edge case where all values are the same
        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else 0.0

        # Calculate R�
        r_squared = 1 - (ss_res / ss_tot)

        return float(r_squared)

    def _apply_simplicity_bias(
        self,
        fits: Dict[ComplexityClass, float],
        best_class: ComplexityClass,
        best_r_sq: float,
    ) -> ComplexityClass:
        """
        Apply Occam's Razor: prefer simpler complexity when R� values are close.

        If multiple complexities have similar R� values (within 0.02), prefer
        the simpler one based on a predefined complexity order.

        Args:
            fits: Dictionary of complexity classes to R� scores
            best_class: Currently best complexity class
            best_r_sq: R� score of best class

        Returns:
            Potentially updated complexity class favoring simplicity
        """
        # Define complexity order from simplest to most complex
        complexity_order = [
            ComplexityClass.CONSTANT,
            ComplexityClass.LOGARITHMIC,
            ComplexityClass.LINEAR,
            ComplexityClass.LINEARITHMIC,
            ComplexityClass.QUADRATIC,
            ComplexityClass.CUBIC,
            ComplexityClass.EXPONENTIAL,
            ComplexityClass.FACTORIAL,
        ]

        threshold = 0.02  # R� difference threshold for considering fits "equal"

        # Find all complexities within threshold of best R�
        similar_fits = [
            cls for cls, r_sq in fits.items()
            if abs(r_sq - best_r_sq) <= threshold
        ]

        if not similar_fits:
            return best_class

        # Return the simplest among similar fits
        for complexity in complexity_order:
            if complexity in similar_fits:
                return complexity

        return best_class

    def get_complexity_function(
        self,
        complexity_class: ComplexityClass,
    ) -> Optional[ComplexityFunction]:
        """
        Get the mathematical function for a given complexity class.

        Args:
            complexity_class: The complexity class to get the function for

        Returns:
            The complexity function, or None if not available
        """
        return self.COMPLEXITY_FUNCTIONS.get(complexity_class)


def fit_to_complexity(
    sizes: list[int],
    times: list[float],
    min_r_squared: float = 0.8,
) -> Tuple[ComplexityClass, float]:
    """
    Convenience function to fit data to complexity curves.

    This is a simplified interface to ComplexityFitter for quick analysis.

    Args:
        sizes: List of input sizes
        times: List of execution times
        min_r_squared: Minimum acceptable R� value

    Returns:
        Tuple of (best_complexity_class, r_squared_score)

    Example:
        >>> sizes = [10, 20, 30, 40, 50]
        >>> times = [0.01, 0.02, 0.03, 0.04, 0.05]
        >>> complexity, r_sq = fit_to_complexity(sizes, times)
        >>> print(f"{complexity}: R�={r_sq:.3f}")
        O(n): R�=1.000
    """
    fitter = ComplexityFitter(min_r_squared=min_r_squared)
    sizes_arr = np.array(sizes, dtype=np.float64)
    times_arr = np.array(times, dtype=np.float64)
    return fitter.fit_complexity(sizes_arr, times_arr)


__all__ = [
    "ComplexityFitter",
    "ComplexityFunction",
    "fit_to_complexity",
]
