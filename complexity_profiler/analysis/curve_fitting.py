"""Empirical growth estimates from finite measurements, not proofs of Big-O."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from complexity_profiler.algorithms.base import ComplexityClass

ComplexityFunction = Callable[[npt.NDArray[np.float64], float, float], npt.NDArray[np.float64]]


class InconclusiveFitError(RuntimeError):
    """No sufficiently good, distinguishable candidate; scores remain inspectable."""

    def __init__(self, reason: str, candidate_scores: Dict[ComplexityClass, float]) -> None:
        super().__init__(f"Inconclusive empirical fit: {reason}")
        self.candidate_scores = dict(candidate_scores)


class ComplexityFitter:
    """Fit nonnegative growth plus an intercept to six candidate curves.

    Successful calls retain the ``(ComplexityClass, R²)`` API. Poor or ambiguous
    fits raise InconclusiveFitError (a RuntimeError) with candidate_scores.
    Candidates within 0.02 R² and twice the best residual error are considered
    indistinguishable (with a floating-point floor for exact fits).
    Exactly constant, positive measurements are reported as a constant fit.
    """

    COMPLEXITY_FUNCTIONS: Dict[ComplexityClass, ComplexityFunction] = {
        ComplexityClass.CONSTANT: lambda n, a, b: a * np.ones_like(n) + b,
        ComplexityClass.LOGARITHMIC: lambda n, a, b: a * np.log(n) + b,
        ComplexityClass.LINEAR: lambda n, a, b: a * n + b,
        ComplexityClass.LINEARITHMIC: lambda n, a, b: a * n * np.log(n) + b,
        ComplexityClass.QUADRATIC: lambda n, a, b: a * n**2 + b,
        ComplexityClass.CUBIC: lambda n, a, b: a * n**3 + b,
    }

    def __init__(self, min_r_squared: float = 0.8) -> None:
        if not np.isfinite(min_r_squared) or not 0 <= min_r_squared <= 1:
            raise ValueError("min_r_squared must be finite and between 0 and 1")
        self.min_r_squared = min_r_squared
        self.candidate_scores: Dict[ComplexityClass, float] = {}

    def fit_complexity(
        self,
        sizes: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> Tuple[ComplexityClass, float]:
        """Return an identifiable empirical fit; reject invalid or weak evidence.

        Inputs must be finite one-dimensional equal-length arrays, with at least
        three distinct positive sizes and nonnegative, nonzero timing data.
        Timing units do not affect the scores. ValueError denotes invalid input;
        InconclusiveFitError denotes valid data without a distinguishable fit.
        """
        self.candidate_scores = {}
        sizes = np.asarray(sizes, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if sizes.ndim != 1 or times.ndim != 1:
            raise ValueError("sizes and times must be one-dimensional")
        if sizes.size != times.size:
            raise ValueError("sizes and times must have the same length")
        if sizes.size < 3:
            raise ValueError("Need at least 3 data points for curve fitting")
        if not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(times)):
            raise ValueError("sizes and times must be finite")
        if np.any(sizes <= 0) or np.any(times < 0) or not np.any(times > 0):
            raise ValueError("sizes must be positive and times nonnegative and nonzero")
        if np.unique(sizes).size != sizes.size:
            raise ValueError("sizes must be distinct")

        # All supported curves are affine in their coefficients. Normalized
        # one-variable least squares avoids nonlinear optimizer tolerances tied
        # to seconds, covariance warnings, and overflow/underflow in residuals.
        normalized_times = times / np.max(times)
        # A positive rescaling of a basis preserves the fitted affine model.
        # Scale n before powers so finite sizes cannot overflow n**2/n**3.
        # Keep log(original n) in n log n: log(n/max(n)) adds a linear term.
        relative_sizes = sizes / np.max(sizes)
        stable_bases = {
            ComplexityClass.CONSTANT: np.ones_like(sizes),
            ComplexityClass.LOGARITHMIC: np.log(sizes),
            ComplexityClass.LINEAR: relative_sizes,
            ComplexityClass.LINEARITHMIC: relative_sizes * np.log(sizes),
            ComplexityClass.QUADRATIC: relative_sizes**2,
            ComplexityClass.CUBIC: relative_sizes**3,
        }
        for complexity, func in self.COMPLEXITY_FUNCTIONS.items():
            self.candidate_scores[complexity] = self._fit_single_complexity(
                sizes, normalized_times, func, basis=stable_bases.get(complexity)
            )
        scores = self.candidate_scores
        best_class = max(scores, key=lambda cls: scores[cls])
        best_score = scores[best_class]
        if best_score < self.min_r_squared:
            raise InconclusiveFitError("no candidate meets min_r_squared", scores)
        if np.all(times == times[0]):
            return ComplexityClass.CONSTANT, scores[ComplexityClass.CONSTANT]
        residual_limit = max(2 * (1 - best_score), 64 * np.finfo(float).eps)
        similar = [
            c
            for c, score in scores.items()
            if best_score - score <= 0.02 and 1 - score <= residual_limit
        ]
        if len(similar) > 1:
            raise InconclusiveFitError("ambiguous candidate scores", scores)
        return best_class, best_score

    def _fit_single_complexity(
        self,
        sizes: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        func: ComplexityFunction,
        *,
        basis: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        # The public curve functions retain their original unscaled coefficients.
        # Callers supplying a raw/custom curve still get an explicit numerical
        # failure, rather than a fabricated zero score that favors another class.
        if basis is None:
            with np.errstate(over="ignore", invalid="ignore"):
                basis = func(sizes, 1.0, 0.0)
        if not np.all(np.isfinite(basis)):
            raise InconclusiveFitError(
                "candidate basis is not representable", self.candidate_scores
            )
        magnitude = float(np.max(np.abs(basis)))
        if magnitude > 0:
            basis = basis / magnitude
        centered = basis - np.mean(basis)
        variance = float(np.dot(centered, centered))
        mean = float(np.mean(times))
        if variance == 0:
            predicted = np.full_like(times, mean)
        else:
            slope = max(0.0, float(np.dot(centered, times - mean)) / variance)
            predicted = mean + slope * centered
        return self._calculate_r_squared(times, predicted)

    def _calculate_r_squared(
        self,
        actual: npt.NDArray[np.float64],
        predicted: npt.NDArray[np.float64],
    ) -> float:
        """Scale-free R²; exact constant data use 1 for equality and 0 otherwise."""
        scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(predicted))))
        if scale == 0:
            return 1.0
        actual = actual / scale
        predicted = predicted / scale
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def _apply_simplicity_bias(
        self,
        fits: Dict[ComplexityClass, float],
        best_class: ComplexityClass,
        best_r_sq: float,
    ) -> ComplexityClass:
        """Legacy helper: never prefer a candidate below the configured threshold."""
        for cls in self.COMPLEXITY_FUNCTIONS:
            if cls in fits and fits[cls] >= self.min_r_squared and best_r_sq - fits[cls] <= 0.02:
                return cls
        return best_class

    def get_complexity_function(
        self, complexity_class: ComplexityClass
    ) -> Optional[ComplexityFunction]:
        """Return the supported curve, or None for an unsupported candidate."""
        return self.COMPLEXITY_FUNCTIONS.get(complexity_class)


def fit_to_complexity(
    sizes: list[int], times: list[float], min_r_squared: float = 0.8
) -> Tuple[ComplexityClass, float]:
    """Convenience tuple API; inconclusive measurements raise InconclusiveFitError."""
    return ComplexityFitter(min_r_squared).fit_complexity(
        np.asarray(sizes, dtype=np.float64), np.asarray(times, dtype=np.float64)
    )


__all__ = ["ComplexityFitter", "ComplexityFunction", "InconclusiveFitError", "fit_to_complexity"]
