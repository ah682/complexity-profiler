"""
Main CLI entry point for Big-O Complexity Analyzer.

Provides commands for analyzing algorithms, comparing performance,
and visualizing complexity curves.
"""

import click
from rich.console import Console
from rich.table import Table
from typing import Optional
from pathlib import Path

from complexity_profiler.algorithms.sorting import (
    MergeSort, QuickSort, SelectionSort, BubbleSort,
    InsertionSort, HeapSort
)
from complexity_profiler.algorithms.searching import (
    LinearSearch, BinarySearch, JumpSearch, InterpolationSearch
)
from complexity_profiler.analysis.profiler import AlgorithmProfiler
from complexity_profiler.data.generators import get_data_generator
from complexity_profiler.visualization.charts import ChartGenerator
from complexity_profiler.visualization.formatters import format_profile_result
from complexity_profiler.export.json_exporter import export_to_json
from complexity_profiler.export.csv_exporter import export_to_csv
from complexity_profiler.utils.exceptions import BigOComplexityError
from complexity_profiler.utils.logging_config import setup_logging

console = Console()

# Algorithm registry
ALGORITHMS = {
    # Sorting algorithms
    "merge_sort": MergeSort(),
    "quick_sort": QuickSort(),
    "selection_sort": SelectionSort(),
    "bubble_sort": BubbleSort(),
    "insertion_sort": InsertionSort(),
    "heap_sort": HeapSort(),
    # Searching algorithms
    "linear_search": LinearSearch(),
    "binary_search": BinarySearch(),
    "jump_search": JumpSearch(),
    "interpolation_search": InterpolationSearch(),
}


@click.group()
@click.version_option(version="1.0.0", prog_name="Big-O Complexity Analyzer")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-file", type=click.Path(), help="Path to log file")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, log_file: Optional[str]) -> None:
    """
    Big-O Complexity Analyzer - Analyze algorithm performance empirically.

    Measure actual complexity, generate visualizations, and export results.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, log_file=log_file)


@cli.command()
@click.argument("algorithm", type=click.Choice(list(ALGORITHMS.keys())))
@click.option(
    "--sizes", "-s",
    default="100,500,1000,5000",
    help="Comma-separated input sizes (default: 100,500,1000,5000)"
)
@click.option(
    "--runs", "-r",
    default=10,
    type=int,
    help="Number of runs per size (default: 10)"
)
@click.option(
    "--data-type", "-d",
    type=click.Choice(["random", "sorted", "reverse", "nearly_sorted"]),
    default="random",
    help="Type of input data (default: random)"
)
@click.option(
    "--export", "-e",
    type=click.Choice(["json", "csv"]),
    help="Export format (json or csv)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path for export"
)
@click.option(
    "--visualize/--no-visualize",
    default=True,
    help="Generate visualization chart (default: yes)"
)
@click.option(
    "--save-chart", "-c",
    type=click.Path(),
    help="Save chart to file instead of displaying"
)
@click.pass_context
def analyze(
    ctx: click.Context,
    algorithm: str,
    sizes: str,
    runs: int,
    data_type: str,
    export: Optional[str],
    output: Optional[str],
    visualize: bool,
    save_chart: Optional[str]
) -> None:
    """
    Analyze a single algorithm's complexity.

    Examples:

        \b
        # Analyze merge sort with default settings
        complexity_profiler analyze merge_sort

        \b
        # Custom sizes and runs
        complexity_profiler analyze quick_sort --sizes 100,1000,10000 --runs 20

        \b
        # Export results to JSON
        complexity_profiler analyze bubble_sort --export json --output results.json

        \b
        # Save visualization chart
        complexity_profiler analyze heap_sort --save-chart heap_sort.png
    """
    try:
        # Parse sizes
        size_list = [int(s.strip()) for s in sizes.split(",")]

        if ctx.obj.get("verbose"):
            console.print(f"[bold blue]Analyzing {algorithm}...[/bold blue]")
            console.print(f"Sizes: {size_list}")
            console.print(f"Runs per size: {runs}")
            console.print(f"Data type: {data_type}")

        # Get algorithm instance
        algo = ALGORITHMS[algorithm]

        # Get data generator
        data_gen = get_data_generator(data_type)

        # Create and configure profiler
        with console.status(f"[bold green]Profiling {algorithm}...", spinner="dots"):
            profiler = AlgorithmProfiler() \
                .with_sizes(size_list) \
                .with_runs(runs) \
                .with_data_generator(data_gen)

            result = profiler.profile(algo)

        # Display results
        console.print("\n")
        console.print(format_profile_result(result))

        # Export if requested
        if export and output:
            output_path = Path(output)
            if export == "json":
                export_to_json(result, output_path)
                console.print(f"\n[green] Results exported to {output_path}[/green]")
            elif export == "csv":
                export_to_csv(result, output_path)
                console.print(f"\n[green] Results exported to {output_path}[/green]")

        # Visualize if requested
        if visualize:
            chart_gen = ChartGenerator()
            if save_chart:
                chart_gen.plot_complexity_curve(result, save_path=save_chart)
                console.print(f"\n[green] Chart saved to {save_chart}[/green]")
            else:
                chart_gen.plot_complexity_curve(result)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise click.Abort()


@cli.command()
@click.argument("algorithms", nargs=-1, required=True)
@click.option("--sizes", "-s", default="100,500,1000,5000")
@click.option("--runs", "-r", default=10, type=int)
@click.option("--data-type", "-d", type=click.Choice(["random", "sorted", "reverse", "nearly_sorted"]), default="random")
@click.option("--save-chart", "-c", type=click.Path(), help="Save comparison chart to file")
@click.pass_context
def compare(
    ctx: click.Context,
    algorithms: tuple[str, ...],
    sizes: str,
    runs: int,
    data_type: str,
    save_chart: Optional[str]
) -> None:
    """
    Compare multiple algorithms.

    Examples:

        \b
        # Compare sorting algorithms
        complexity_profiler compare merge_sort quick_sort heap_sort

        \b
        # Compare with custom settings
        complexity_profiler compare bubble_sort insertion_sort --sizes 100,500,1000 --runs 5
    """
    try:
        # Validate algorithms
        invalid = [a for a in algorithms if a not in ALGORITHMS]
        if invalid:
            console.print(f"[red]Invalid algorithms: {', '.join(invalid)}[/red]")
            console.print(f"Available: {', '.join(ALGORITHMS.keys())}")
            raise click.Abort()

        size_list = [int(s.strip()) for s in sizes.split(",")]
        data_gen = get_data_generator(data_type)

        results = []

        # Profile each algorithm
        for algo_name in algorithms:
            algo = ALGORITHMS[algo_name]

            with console.status(f"[bold green]Profiling {algo_name}...", spinner="dots"):
                profiler = AlgorithmProfiler() \
                    .with_sizes(size_list) \
                    .with_runs(runs) \
                    .with_data_generator(data_gen)

                result = profiler.profile(algo)
                results.append(result)

        # Display comparison table
        table = Table(title="Algorithm Comparison")
        table.add_column("Algorithm", style="cyan", no_wrap=True)
        table.add_column("Complexity", style="magenta")
        table.add_column("Fit Quality (R²)", style="green")
        table.add_column("Stable", style="yellow")

        for result in results:
            algo_inst = ALGORITHMS[result.algorithm_name.lower().replace(" ", "_")]
            stable = "Yes" if algo_inst.metadata.stable else "No"
            table.add_row(
                result.algorithm_name,
                str(result.fitted_complexity),
                f"{result.fit_quality:.4f}",
                stable
            )

        console.print("\n")
        console.print(table)

        # Visualize comparison
        chart_gen = ChartGenerator()
        if save_chart:
            chart_gen.plot_comparison(results, save_path=save_chart)
            console.print(f"\n[green] Comparison chart saved to {save_chart}[/green]")
        else:
            chart_gen.plot_comparison(results)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise click.Abort()


@cli.command()
@click.option("--list-sorting", is_flag=True, help="List sorting algorithms")
@click.option("--list-searching", is_flag=True, help="List searching algorithms")
def list_algorithms(list_sorting: bool, list_searching: bool) -> None:
    """List available algorithms."""

    if not list_sorting and not list_searching:
        # List all
        list_sorting = list_searching = True

    if list_sorting:
        console.print("\n[bold cyan]Sorting Algorithms:[/bold cyan]")
        table = Table()
        table.add_column("Name", style="green")
        table.add_column("Complexity", style="yellow")
        table.add_column("Stable", style="magenta")

        for name, algo in ALGORITHMS.items():
            if algo.metadata.category == "sorting":
                table.add_row(
                    name,
                    str(algo.metadata.expected_complexity),
                    "Yes" if algo.metadata.stable else "No"
                )
        console.print(table)

    if list_searching:
        console.print("\n[bold cyan]Searching Algorithms:[/bold cyan]")
        table = Table()
        table.add_column("Name", style="green")
        table.add_column("Complexity", style="yellow")

        for name, algo in ALGORITHMS.items():
            if algo.metadata.category == "searching":
                table.add_row(name, str(algo.metadata.expected_complexity))
        console.print(table)


if __name__ == "__main__":
    cli()
