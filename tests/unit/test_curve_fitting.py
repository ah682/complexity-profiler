"""Regressions with independently constructed growth curves and residuals."""

import numpy as np
import pytest

from complexity_profiler.algorithms.base import ComplexityClass
from complexity_profiler.analysis.curve_fitting import ComplexityFitter, fit_to_complexity


def test_microsecond_constant_prediction_has_zero_r_squared():
    actual = np.array([1e-6, 2e-6, 4e-6])
    predicted = np.full(3, 7e-6 / 3)
    assert ComplexityFitter()._calculate_r_squared(actual, predicted) == pytest.approx(0)


@pytest.mark.parametrize("scale", [1e-200, 1e-6, 1, 1e6, 1e200])
@pytest.mark.parametrize(
    "times,expected",
    [
        ([1, 2, 4, 8, 16, 32], ComplexityClass.LINEAR),
        ([1, 4, 16, 64, 256, 1024], ComplexityClass.QUADRATIC),
        ([1, 8, 64, 512, 4096, 32768], ComplexityClass.CUBIC),
        ([2, 2, 2, 2, 2, 2], ComplexityClass.CONSTANT),
        ([0, 1, 2, 3, 4, 5], ComplexityClass.LOGARITHMIC),
        ([0, 2, 8, 24, 64, 160], ComplexityClass.LINEARITHMIC),
    ],
)
def test_known_growth_is_independent_of_timing_units(scale, times, expected):
    cls, score = fit_to_complexity([1, 2, 4, 8, 16, 32], [scale * t for t in times])
    assert cls == expected
    assert score == pytest.approx(1)


@pytest.mark.parametrize(
    "sizes,times",
    [
        ([1, 2, 3], [1, float("nan"), 3]),
        ([1, float("inf"), 3], [1, 2, 3]),
        ([0, 2, 3], [1, 2, 3]),
        ([-1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, -2, 3]),
        ([1, 1, 1], [1, 2, 3]),
        ([1, 1, 2], [1, 2, 3]),
        ([[1], [2], [3]], [1, 2, 3]),
        ([1, 2, 3], [[1], [2], [3]]),
        ([1, 2, 3], [0, 0, 0]),
        ([1, 2], [1, 2]),
        ([1, 2, 3], [1, 2]),
    ],
)
def test_invalid_or_unidentifiable_measurements_are_rejected(sizes, times):
    with pytest.raises(ValueError):
        fit_to_complexity(sizes, times)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_minimum_score_is_rejected(threshold):
    with pytest.raises(ValueError):
        ComplexityFitter(min_r_squared=threshold)


def test_minimum_fit_quality_is_enforced_and_exposes_candidates():
    fitter = ComplexityFitter(min_r_squared=0.99)
    with pytest.raises(RuntimeError, match="Inconclusive") as exc:
        fitter.fit_complexity(np.array([1, 2, 4, 8, 16, 32]), np.array([5, 1, 9, 2, 8, 3]))
    assert max(exc.value.candidate_scores.values()) < 0.99
    assert len(exc.value.candidate_scores) == 6


def test_decreasing_times_cannot_claim_positive_growth():
    with pytest.raises(RuntimeError, match="Inconclusive"):
        fit_to_complexity([1, 2, 4, 8, 16, 32], [32, 16, 8, 4, 2, 1])


def test_narrow_range_with_indistinguishable_candidates_is_inconclusive():
    with pytest.raises(RuntimeError, match="ambiguous"):
        fit_to_complexity([100, 101, 102, 103], [100, 101.02, 101.98, 103])


def test_simplicity_cannot_select_candidate_below_threshold():
    fitter = ComplexityFitter(min_r_squared=0.995)
    scores = {ComplexityClass.LINEAR: 0.98, ComplexityClass.QUADRATIC: 0.999}
    assert (
        fitter._apply_simplicity_bias(scores, ComplexityClass.QUADRATIC, 0.999)
        == ComplexityClass.QUADRATIC
    )


def test_cubic_growth_with_large_finite_integer_sizes_does_not_overflow():
    sizes = [10**110 * 2**k for k in range(6)]
    times = [float(8**k) for k in range(6)]
    cls, score = fit_to_complexity(sizes, times)
    assert cls == ComplexityClass.CUBIC
    assert score == pytest.approx(1)


def test_linear_growth_with_tiny_finite_sizes_does_not_underflow():
    sizes = np.array([1, 2, 4, 8, 16, 32], dtype=float) * 1e-200
    times = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    cls, score = ComplexityFitter().fit_complexity(sizes, times)
    assert cls == ComplexityClass.LINEAR
    assert score == pytest.approx(1)


def test_unrepresentable_raw_basis_is_inconclusive_instead_of_a_zero_score():
    from complexity_profiler.analysis.curve_fitting import InconclusiveFitError

    fitter = ComplexityFitter()
    cubic = fitter.get_complexity_function(ComplexityClass.CUBIC)
    assert cubic is not None
    with pytest.raises(InconclusiveFitError, match="basis"):
        fitter._fit_single_complexity(
            np.array([1e110, 2e110, 4e110]), np.array([1.0, 8.0, 64.0]), cubic
        )


def test_linearithmic_fit_retains_original_size_logarithm():
    sizes = np.array([100, 200, 400, 800, 1600, 3200], dtype=float)
    times = 2.0 * sizes * np.log(sizes) + 17.0
    cls, score = ComplexityFitter().fit_complexity(sizes, times)
    assert cls == ComplexityClass.LINEARITHMIC
    assert score == pytest.approx(1)


def test_public_curve_functions_keep_unscaled_coefficients():
    fitter = ComplexityFitter()
    cubic = fitter.get_complexity_function(ComplexityClass.CUBIC)
    linearithmic = fitter.get_complexity_function(ComplexityClass.LINEARITHMIC)
    assert cubic is not None and linearithmic is not None
    np.testing.assert_allclose(cubic(np.array([1.0, 2.0, 4.0]), 2.0, 3.0), [5.0, 19.0, 131.0])
    np.testing.assert_allclose(
        linearithmic(np.array([1.0, 2.0, 4.0]), 2.0, 3.0),
        [3.0, 3.0 + 4.0 * np.log(2.0), 3.0 + 8.0 * np.log(4.0)],
    )
