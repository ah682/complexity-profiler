"""Analysis components for profiling and complexity fitting."""

from complexity_profiler.analysis.curve_fitting import (
    ComplexityFitter,
    ComplexityFunction,
    fit_to_complexity,
)
from complexity_profiler.analysis.metrics import (
    AggregateMetricsCollector,
    DefaultMetricsCollector,
    PerformanceMetrics,
)
from complexity_profiler.analysis.profiler import (
    AlgorithmProfiler,
    ProfileResult,
    profile_algorithm,
)
from complexity_profiler.analysis.statistics import Statistics, compute_statistics

__all__ = [
    # Statistics
    "Statistics",
    "compute_statistics",
    # Metrics
    "PerformanceMetrics",
    "DefaultMetricsCollector",
    "AggregateMetricsCollector",
    # Curve Fitting
    "ComplexityFitter",
    "ComplexityFunction",
    "fit_to_complexity",
    # Profiler
    "ProfileResult",
    "AlgorithmProfiler",
    "profile_algorithm",
]
