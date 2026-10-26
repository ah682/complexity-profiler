"""Create a reproducible chart from controlled data, not a timing claim."""

from pathlib import Path

from complexity_profiler.analysis.curve_fitting import fit_to_complexity
from complexity_profiler.analysis.metrics import PerformanceMetrics
from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.analysis.statistics import compute_statistics
from complexity_profiler.export import export_to_json
from complexity_profiler.visualization.charts import ChartGenerator

sizes = [1, 2, 4, 8, 16, 32]
times = [1e-6, 2e-6, 4e-6, 8e-6, 16e-6, 32e-6]
complexity, score = fit_to_complexity(sizes, times)
result = ProfileResult(
    algorithm_name="Controlled linear measurements",
    category="synthetic",
    input_sizes=sizes,
    metrics_per_size={n: [PerformanceMetrics(execution_time=t)] for n, t in zip(sizes, times)},
    statistics_per_size={n: compute_statistics([t]) for n, t in zip(sizes, times)},
    empirical_complexity=str(complexity),
    r_squared=score,
    notes="Literal synthetic measurements; not an algorithm benchmark.",
)
export_to_json(result, Path("controlled-growth.json"))
generator = ChartGenerator()
generator.plot_complexity_curve(result, save_path=Path("controlled-growth.png"))
generator.close()
