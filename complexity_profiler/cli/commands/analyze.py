"""
Analyze command implementation for profiling individual algorithms.

This module implements the 'analyze' CLI command which profiles a single
algorithm and displays or exports the results.
"""

from typing import Optional, List
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from complexity_profiler.analysis.profiler import AlgorithmProfiler
from complexity_profiler.visualization.formatters import ResultFormatter
from complexity_profiler.visualization.charts import ChartGenerator
from complexity_profiler.cli.validators import (
    validate_algorithm_name,
    validate_input_sizes,
    validate_runs,
    validate_data_type,
    validate_export_format,
    validate_output_path,
    validate_metric,
    get_algorithm_instance,
    create_data_generator,
    VALID_METRICS,
)


@click.command(name="analyze")
@click.argument(
    "algorithm",
    type=str,
    callback=validate_algorithm_name,
)
@click.option(
    "--sizes",
    "-s",
    type=str,
    default="100,500,1000,5000",
    callback=validate_input_sizes,
    help="Input sizes to test (comma-separated or range notation like '100-1000:5')",
)
@click.option(
    "--runs",
    "-r",
    type=int,
    default=5,
    callback=validate_runs,
    help="Number of runs per input size for statistical reliability",
)
@click.option(
    "--data-type",
    "-d",
    type=str,
    default="random",
    callback=validate_data_type,
    help="Type of test data: random, sorted, reversed, nearly_sorted",
)
@click.option(
    "--export",
    "-e",
    type=str,
    callback=validate_export_format,
    help="Export results to file (json, csv, html)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    callback=validate_output_path,
    help="Output file path for export",
)
@click.option(
    "--visualize",
    "-v",
    type=str,
    callback=validate_output_path,
    help="Generate and save visualization chart to specified path",
)
@click.option(
    "--metric",
    "-m",
    type=str,
    default="execution_time",
    callback=validate_metric,
    help=f"Metric to visualize: {', '.join(sorted(VALID_METRICS))}",
)
@click.option(
    "--log-scale",
    is_flag=True,
    help="Use log-log scale for visualization",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducible results",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress console output",
)
def analyze(
    algorithm: str,
    sizes: List[int],
    runs: int,
    data_type: str,
    export: Optional[str],
    output: Optional[Path],
    visualize: Optional[Path],
    metric: str,
    log_scale: bool,
    seed: Optional[int],
    quiet: bool,
) -> None:
    """
    Analyze the complexity of a single algorithm.

    Profiles the specified algorithm across different input sizes,
    collects performance metrics, and displays results in a formatted table.
    Optionally exports results and/or generates visualizations.

    ALGORITHM: Name of the algorithm to analyze (e.g., merge_sort, quick_sort)

    \b
    Examples:
        # Basic analysis
        complexity_profiler analyze merge_sort

        # Custom input sizes and runs
        complexity_profiler analyze quick_sort --sizes "100,500,1000,5000" --runs 10

        # Use range notation for sizes
        complexity_profiler analyze bubble_sort --sizes "100-5000:10"

        # Analyze with worst-case data
        complexity_profiler analyze insertion_sort --data-type reversed

        # Export results and visualization
        complexity_profiler analyze heap_sort --export json --output results.json \\
            --visualize chart.png

        # Visualize comparisons with log scale
        complexity_profiler analyze selection_sort --visualize chart.png \\
            --metric comparisons --log-scale
    """
    console = Console()
    formatter = ResultFormatter(console)

    if not quiet:
        console.print(
            f"\n[bold cyan]Analyzing Algorithm:[/bold cyan] [yellow]{algorithm}[/yellow]\n"
        )

    # Load algorithm
    try:
        algo_instance = get_algorithm_instance(algorithm)
    except ValueError as e:
        formatter.print_error(str(e))
        raise click.Abort()

    # Create data generator
    data_gen = create_data_generator(data_type, seed)

    # Display configuration
    if not quiet:
        config_info = [
            f"[bold]Input Sizes:[/] {', '.join(map(str, sizes))}",
            f"[bold]Runs per Size:[/] {runs}",
            f"[bold]Data Type:[/] {data_type}",
            f"[bold]Expected Complexity:[/] {algo_instance.metadata.expected_complexity}",
        ]
        if seed is not None:
            config_info.append(f"[bold]Random Seed:[/] {seed}")

        console.print("\n".join(config_info))
        console.print()

    # Profile the algorithm with progress bar
    result = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                f"Profiling {algorithm}...",
                total=len(sizes) * runs
            )

            # Create profiler
            profiler = AlgorithmProfiler()
            profiler = (profiler
                       .with_sizes(sizes)
                       .with_runs(runs)
                       .with_data_generator(data_gen))

            # Profile with progress updates
            result = profiler.profile(algo_instance)
            progress.update(task, completed=len(sizes) * runs)

    except Exception as e:
        formatter.print_error(f"Profiling failed: {e}")
        raise click.Abort()

    # Display results
    if not quiet:
        console.print()
        console.print(formatter.format_summary_panel(result))
        console.print()
        console.print(formatter.format_results_table(result))
        console.print()

    # Export results if requested
    if export and output:
        try:
            _export_results(result, export, output, formatter, quiet)
        except Exception as e:
            formatter.print_error(f"Export failed: {e}")
            raise click.Abort()

    # Generate visualization if requested
    if visualize:
        try:
            _generate_visualization(
                result,
                visualize,
                metric,
                log_scale,
                formatter,
                quiet
            )
        except Exception as e:
            formatter.print_error(f"Visualization failed: {e}")
            raise click.Abort()

    if not quiet:
        formatter.print_success("Analysis complete!")


def _export_results(
    result,
    format: str,
    output_path: Path,
    formatter: ResultFormatter,
    quiet: bool,
) -> None:
    """
    Export profiling results to file.

    Args:
        result: ProfileResult to export
        format: Export format (json, csv, html)
        output_path: Path to save file
        formatter: ResultFormatter for console output
        quiet: Whether to suppress output
    """
    import json
    import csv
    from datetime import datetime

    if format == "json":
        # Export as JSON
        data = result.to_dict()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "csv":
        # Export as CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "Input Size",
                "Mean Time (s)",
                "Std Dev (s)",
                "Min Time (s)",
                "Max Time (s)",
                "Mean Comparisons",
                "Mean Swaps",
                "Mean Accesses",
            ])

            # Write data rows
            for size in result.input_sizes:
                stats = result.statistics_per_size[size]
                metrics_list = result.metrics_per_size[size]

                mean_comp = sum(m.comparisons for m in metrics_list) / len(metrics_list)
                mean_swaps = sum(m.swaps for m in metrics_list) / len(metrics_list)
                mean_acc = sum(m.accesses for m in metrics_list) / len(metrics_list)

                writer.writerow([
                    size,
                    f"{stats.mean:.6f}",
                    f"{stats.std_dev:.6f}",
                    f"{stats.min:.6f}",
                    f"{stats.max:.6f}",
                    f"{mean_comp:.0f}",
                    f"{mean_swaps:.0f}",
                    f"{mean_acc:.0f}",
                ])

    elif format == "html":
        # Export as HTML
        html_content = _generate_html_report(result)
        with open(output_path, "w") as f:
            f.write(html_content)

    if not quiet:
        formatter.print_success(f"Results exported to: {output_path}")


def _generate_visualization(
    result,
    output_path: Path,
    metric: str,
    log_scale: bool,
    formatter: ResultFormatter,
    quiet: bool,
) -> None:
    """
    Generate and save visualization chart.

    Args:
        result: ProfileResult to visualize
        output_path: Path to save chart
        metric: Metric to plot
        log_scale: Whether to use log scale
        formatter: ResultFormatter for console output
        quiet: Whether to suppress output
    """
    generator = ChartGenerator()

    # Generate chart
    generator.plot_complexity_curve(
        result,
        metric=metric,
        log_scale=log_scale,
        show_error_bars=True,
    )

    # Save chart
    generator.save(output_path)
    generator.close()

    if not quiet:
        formatter.print_success(f"Visualization saved to: {output_path}")


def _generate_html_report(result) -> str:
    """
    Generate HTML report for profiling results.

    Args:
        result: ProfileResult to convert to HTML

    Returns:
        HTML string
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{result.algorithm_name} - Profiling Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .complexity {{
            color: #e74c3c;
            font-weight: bold;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <h1>{result.algorithm_name} - Profiling Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Algorithm:</strong> {result.algorithm_name}</p>
        <p><strong>Category:</strong> {result.category}</p>
        <p><strong>Detected Complexity:</strong> <span class="complexity">{result.empirical_complexity}</span></p>
        <p><strong>R² Score:</strong> {result.r_squared:.4f}</p>
        <p><strong>Timestamp:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>Performance Data</h2>
    <table>
        <thead>
            <tr>
                <th>Input Size</th>
                <th>Mean Time (s)</th>
                <th>Std Dev (s)</th>
                <th>Min (s)</th>
                <th>Max (s)</th>
            </tr>
        </thead>
        <tbody>
"""

    for size in result.input_sizes:
        stats = result.statistics_per_size[size]
        html += f"""
            <tr>
                <td>{size:,}</td>
                <td>{stats.mean:.6f}</td>
                <td>{stats.std_dev:.6f}</td>
                <td>{stats.min:.6f}</td>
                <td>{stats.max:.6f}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
</body>
</html>
"""

    return html


__all__ = ["analyze"]
