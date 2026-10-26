"""Fit evidence survives the profiler and export boundary."""

import pytest

from complexity_profiler.algorithms.sorting import MergeSort
from complexity_profiler.analysis.metrics import PerformanceMetrics
from complexity_profiler.analysis.profiler import AlgorithmProfiler
from complexity_profiler.export import export_to_json, load_from_json


def test_inconclusive_profile_retains_measurements_and_scores(monkeypatch, tmp_path):
    # Poor, hand-chosen zig-zag data must not acquire a complexity label.
    times = dict(zip([1, 2, 4, 8, 16, 32], [5, 1, 9, 2, 8, 3]))
    monkeypatch.setattr(
        AlgorithmProfiler,
        "_profile_single_size",
        lambda self, algorithm, size: [PerformanceMetrics(execution_time=times[size])],
    )
    result = (
        AlgorithmProfiler()
        .with_sizes(list(times))
        .with_runs(1)
        .with_data_generator(lambda n: list(range(n)))
        .profile(MergeSort())
    )
    assert result.empirical_complexity == "Inconclusive"
    assert result.notes and "min_r_squared" in result.notes
    assert result.statistics_per_size[4].mean == 9
    assert len(result.candidate_scores) == 6
    path = tmp_path / "profile.json"
    export_to_json(result, path)
    loaded = load_from_json(path)
    assert loaded.candidate_scores == result.candidate_scores
    assert loaded.empirical_complexity == "Inconclusive"


@pytest.mark.parametrize(
    "times,expected",
    [
        ([1, 2, 4, 8, 16, 32], "O(n)"),
        ([1, 4, 16, 64, 256, 1024], "O(n^2)"),
        ([1, 8, 64, 512, 4096, 32768], "O(n^3)"),
        ([1, 1, 1, 1, 1, 1], "O(1)"),
        ([1, 2, 3, 4, 5, 6], "O(log n)"),
        ([2, 8, 24, 64, 160, 384], "O(n log n)"),
    ],
)
def test_pipeline_classifies_controlled_measurements(monkeypatch, times, expected):
    # n=2,4,...64; log2(n)=1,...6, n*log2(n)=2,8,...384.
    observations = dict(zip([2, 4, 8, 16, 32, 64], times))
    monkeypatch.setattr(
        AlgorithmProfiler,
        "_profile_single_size",
        lambda self, algorithm, size: [
            PerformanceMetrics(execution_time=observations[size] * 1e-6)
        ],
    )
    result = (
        AlgorithmProfiler()
        .with_sizes(list(observations))
        .with_runs(1)
        .with_data_generator(lambda n: list(range(n)))
        .profile(MergeSort())
    )
    assert result.empirical_complexity == expected
    assert result.r_squared == pytest.approx(1)
    assert result.candidate_scores[expected] == pytest.approx(1)
