"""
Example demonstrating the export functionality for Big-O Complexity Analyzer.

This example shows how to use the JSON and CSV export functions to save
profiling results to different formats.
"""

from datetime import datetime
from pathlib import Path

from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.analysis.metrics import PerformanceMetrics
from complexity_profiler.analysis.statistics import Statistics, compute_statistics
from complexity_profiler.export import export_to_json, export_to_csv, export_metrics_summary_csv


def create_sample_profile_result() -> ProfileResult:
    """
    Create a sample ProfileResult for demonstration.

    Returns:
        A complete ProfileResult with sample data
    """
    # Sample input sizes and metrics
    input_sizes = [100, 500, 1000, 5000, 10000]

    # Create metrics for each input size
    metrics_per_size = {}
    statistics_per_size = {}

    for size in input_sizes:
        # Simulate 3 runs per input size
        metrics_list = []
        execution_times = []

        for run in range(3):
            # Create metrics with values proportional to input size
            metrics = PerformanceMetrics(
                comparisons=int(size * (run + 1) * 0.5),
                swaps=int(size * (run + 1) * 0.3),
                accesses=int(size * (run + 1) * 0.7),
                recursive_calls=int(size * 0.05 * (run + 1)),
                execution_time=size * 0.001 * (run + 1),
                memory_operations=int(size * 0.1 * (run + 1)),
            )
            metrics_list.append(metrics)
            execution_times.append(metrics.execution_time)

        # Store metrics for this size
        metrics_per_size[size] = metrics_list

        # Compute statistics from execution times
        stats = compute_statistics(execution_times)
        statistics_per_size[size] = stats

    # Create the ProfileResult
    result = ProfileResult(
        algorithm_name="QuickSort",
        category="sorting",
        input_sizes=input_sizes,
        metrics_per_size=metrics_per_size,
        statistics_per_size=statistics_per_size,
        empirical_complexity="O(n log n)",
        r_squared=0.98,
        timestamp=datetime.now(),
        notes="Sample profiling run for demonstration purposes",
    )

    return result


def main() -> None:
    """Run the export example."""
    # Create sample data
    result = create_sample_profile_result()

    # Create output directory
    output_dir = Path("export_results")
    output_dir.mkdir(exist_ok=True)

    # Export to JSON
    json_path = output_dir / "quicksort_results.json"
    print(f"Exporting to JSON: {json_path}")
    export_to_json(result, json_path, pretty=True)
    print(f"  ✓ JSON export successful")

    # Export to CSV
    csv_path = output_dir / "quicksort_results.csv"
    print(f"Exporting to CSV: {csv_path}")
    export_to_csv(result, csv_path)
    print(f"  ✓ CSV export successful")

    # Export summary to CSV
    summary_csv_path = output_dir / "quicksort_summary.csv"
    print(f"Exporting summary to CSV: {summary_csv_path}")
    export_metrics_summary_csv(result, summary_csv_path)
    print(f"  ✓ Summary CSV export successful")

    # Print summary information
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    summary = result.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("INPUT SIZES TESTED")
    print("=" * 60)
    for size in result.input_sizes:
        metrics = result.metrics_per_size[size]
        stats = result.statistics_per_size[size]
        print(f"\nInput Size: {size}")
        print(f"  Runs: {len(metrics)}")
        print(f"  Mean Execution Time: {stats.mean:.6f}s")
        print(f"  Median Execution Time: {stats.median:.6f}s")
        print(f"  Std Dev: {stats.std_dev:.6f}s")
        print(f"  Min: {stats.min:.6f}s")
        print(f"  Max: {stats.max:.6f}s")
        print(f"  Coefficient of Variation: {stats.coefficient_of_variation:.4f}")
        print(f"  Consistent: {stats.is_consistent()}")


if __name__ == "__main__":
    main()
