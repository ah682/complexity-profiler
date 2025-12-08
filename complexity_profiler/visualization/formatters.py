"""
Display formatters for terminal output using Rich library.

This module provides beautiful, formatted console output for profiling results,
metrics, and algorithm comparisons using the Rich library for enhanced terminal
displays.
"""

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import box

from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.analysis.metrics import PerformanceMetrics
from complexity_profiler.analysis.statistics import Statistics
from complexity_profiler.visualization.themes import CONSOLE_THEME


class ResultFormatter:
    """
    Formatter for creating beautiful console displays of profiling results.

    Uses Rich library to create tables, panels, and formatted text for
    professional terminal output.

    Example:
        >>> formatter = ResultFormatter()
        >>> table = formatter.format_results_table(result)
        >>> console = Console()
        >>> console.print(table)
    """

    def __init__(self, console: Console = None) -> None:
        """
        Initialize the formatter.

        Args:
            console: Optional Rich Console instance. Creates new one if None.
        """
        self.console = console or Console()

    def format_results_table(self, result: ProfileResult) -> Table:
        """
        Create a formatted table displaying profiling results.

        Includes input sizes, execution times, and key metrics with
        statistical information.

        Args:
            result: ProfileResult to format

        Returns:
            Rich Table object ready for printing

        Example:
            >>> table = formatter.format_results_table(result)
            >>> console.print(table)
        """
        table = Table(
            title=f"[bold cyan]{result.algorithm_name} - Performance Profile[/]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )

        # Add columns
        table.add_column("Input Size", justify="right", style="yellow", width=12)
        table.add_column("Time (s)", justify="right", style="green", width=15)
        table.add_column("Comparisons", justify="right", style="blue", width=15)
        table.add_column("Swaps", justify="right", style="magenta", width=15)
        table.add_column("Accesses", justify="right", style="cyan", width=15)

        # Add rows for each input size
        for size in result.input_sizes:
            stats = result.statistics_per_size[size]
            metrics_list = result.metrics_per_size[size]

            # Calculate means for other metrics
            mean_comparisons = sum(m.comparisons for m in metrics_list) / len(metrics_list)
            mean_swaps = sum(m.swaps for m in metrics_list) / len(metrics_list)
            mean_accesses = sum(m.accesses for m in metrics_list) / len(metrics_list)

            table.add_row(
                f"{size:,}",
                f"{stats.mean:.6f} ± {stats.std_dev:.6f}",
                f"{mean_comparisons:,.0f}",
                f"{mean_swaps:,.0f}",
                f"{mean_accesses:,.0f}",
            )

        return table

    def format_summary_panel(self, result: ProfileResult) -> Panel:
        """
        Create a formatted panel with summary information.

        Displays algorithm metadata, detected complexity, and quality metrics.

        Args:
            result: ProfileResult to summarize

        Returns:
            Rich Panel object

        Example:
            >>> panel = formatter.format_summary_panel(result)
            >>> console.print(panel)
        """
        summary_lines = [
            f"[bold]Algorithm:[/] [{CONSOLE_THEME['algorithm_name']}]{result.algorithm_name}[/]",
            f"[bold]Category:[/] [{CONSOLE_THEME['info']}]{result.category}[/]",
            f"[bold]Detected Complexity:[/] [{CONSOLE_THEME['complexity_class']}]{result.empirical_complexity}[/]",
            f"[bold]R² Score:[/] [{CONSOLE_THEME['metric_value']}]{result.r_squared:.4f}[/]",
            f"[bold]Input Sizes:[/] {self._format_list(result.input_sizes)}",
            f"[bold]Timestamp:[/] {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        content = "\n".join(summary_lines)

        return Panel(
            content,
            title="[bold cyan]Profiling Summary[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def format_comparison_table(self, results: List[ProfileResult]) -> Table:
        """
        Create a comparison table for multiple algorithms.

        Shows side-by-side comparison of complexity, performance, and metrics.

        Args:
            results: List of ProfileResult objects to compare

        Returns:
            Rich Table object

        Example:
            >>> table = formatter.format_comparison_table([result1, result2])
            >>> console.print(table)
        """
        table = Table(
            title="[bold cyan]Algorithm Comparison[/]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )

        # Add columns
        table.add_column("Algorithm", style="yellow", width=20)
        table.add_column("Complexity", style="magenta", width=15)
        table.add_column("R² Score", justify="right", style="green", width=10)
        table.add_column("Avg Time (s)", justify="right", style="blue", width=15)
        table.add_column("Input Sizes", style="cyan", width=20)

        # Add row for each algorithm
        for result in results:
            # Calculate average execution time across all sizes
            avg_time = sum(
                result.statistics_per_size[size].mean
                for size in result.input_sizes
            ) / len(result.input_sizes)

            table.add_row(
                result.algorithm_name,
                result.empirical_complexity,
                f"{result.r_squared:.4f}",
                f"{avg_time:.6f}",
                self._format_list(result.input_sizes, max_items=3),
            )

        return table

    def format_metrics(self, metrics: PerformanceMetrics) -> str:
        """
        Format a PerformanceMetrics object as a readable string.

        Args:
            metrics: PerformanceMetrics to format

        Returns:
            Formatted string with all metrics

        Example:
            >>> formatted = formatter.format_metrics(metrics)
            >>> print(formatted)
        """
        lines = [
            f"[{CONSOLE_THEME['metric_label']}]Execution Time:[/] "
            f"[{CONSOLE_THEME['metric_value']}]{metrics.execution_time:.6f}s[/]",
            f"[{CONSOLE_THEME['metric_label']}]Comparisons:[/] "
            f"[{CONSOLE_THEME['metric_value']}]{metrics.comparisons:,}[/]",
            f"[{CONSOLE_THEME['metric_label']}]Swaps:[/] "
            f"[{CONSOLE_THEME['metric_value']}]{metrics.swaps:,}[/]",
            f"[{CONSOLE_THEME['metric_label']}]Accesses:[/] "
            f"[{CONSOLE_THEME['metric_value']}]{metrics.accesses:,}[/]",
            f"[{CONSOLE_THEME['metric_label']}]Recursive Calls:[/] "
            f"[{CONSOLE_THEME['metric_value']}]{metrics.recursive_calls:,}[/]",
            f"[{CONSOLE_THEME['metric_label']}]Total Operations:[/] "
            f"[{CONSOLE_THEME['metric_value']}]{metrics.total_operations():,}[/]",
        ]
        return "\n".join(lines)

    def format_statistics(self, stats: Statistics, label: str = "Statistics") -> Panel:
        """
        Format Statistics object as a panel.

        Args:
            stats: Statistics object to format
            label: Label for the panel title

        Returns:
            Rich Panel with formatted statistics

        Example:
            >>> panel = formatter.format_statistics(stats, "Execution Time")
            >>> console.print(panel)
        """
        lines = [
            f"[bold]Mean:[/] [{CONSOLE_THEME['metric_value']}]{stats.mean:.6f}[/]",
            f"[bold]Median:[/] [{CONSOLE_THEME['metric_value']}]{stats.median:.6f}[/]",
            f"[bold]Std Dev:[/] [{CONSOLE_THEME['metric_value']}]{stats.std_dev:.6f}[/]",
            f"[bold]Min:[/] [{CONSOLE_THEME['metric_value']}]{stats.min:.6f}[/]",
            f"[bold]Max:[/] [{CONSOLE_THEME['metric_value']}]{stats.max:.6f}[/]",
            f"[bold]25th %ile:[/] [{CONSOLE_THEME['metric_value']}]{stats.percentile_25:.6f}[/]",
            f"[bold]75th %ile:[/] [{CONSOLE_THEME['metric_value']}]{stats.percentile_75:.6f}[/]",
            f"[bold]CV:[/] [{CONSOLE_THEME['metric_value']}]{stats.coefficient_of_variation:.4f}[/]",
        ]

        consistency = "[green]Consistent[/]" if stats.is_consistent() else "[yellow]Variable[/]"
        lines.append(f"[bold]Consistency:[/] {consistency}")

        return Panel(
            "\n".join(lines),
            title=f"[bold cyan]{label}[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def format_tree_view(self, results: List[ProfileResult]) -> Tree:
        """
        Create a tree view of profiling results.

        Useful for hierarchical display of multiple algorithm results.

        Args:
            results: List of ProfileResult objects

        Returns:
            Rich Tree object

        Example:
            >>> tree = formatter.format_tree_view([result1, result2])
            >>> console.print(tree)
        """
        tree = Tree(
            f"[bold cyan]Profiling Results ({len(results)} algorithms)[/]",
            guide_style="cyan",
        )

        for result in results:
            algo_branch = tree.add(
                f"[{CONSOLE_THEME['algorithm_name']}]{result.algorithm_name}[/] "
                f"- [{CONSOLE_THEME['complexity_class']}]{result.empirical_complexity}[/]"
            )

            algo_branch.add(f"Category: [{CONSOLE_THEME['info']}]{result.category}[/]")
            algo_branch.add(f"R² Score: [{CONSOLE_THEME['metric_value']}]{result.r_squared:.4f}[/]")

            sizes_branch = algo_branch.add("Input Sizes")
            for size in result.input_sizes:
                stats = result.statistics_per_size[size]
                sizes_branch.add(
                    f"n={size:,}: {stats.mean:.6f}s ± {stats.std_dev:.6f}s"
                )

        return tree

    def print_success(self, message: str) -> None:
        """
        Print a success message.

        Args:
            message: Success message to display
        """
        self.console.print(f"[{CONSOLE_THEME['success']}]✓ {message}[/]")

    def print_error(self, message: str) -> None:
        """
        Print an error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[{CONSOLE_THEME['error']}]✗ {message}[/]")

    def print_warning(self, message: str) -> None:
        """
        Print a warning message.

        Args:
            message: Warning message to display
        """
        self.console.print(f"[{CONSOLE_THEME['warning']}]⚠ {message}[/]")

    def print_info(self, message: str) -> None:
        """
        Print an info message.

        Args:
            message: Info message to display
        """
        self.console.print(f"[{CONSOLE_THEME['info']}]ℹ {message}[/]")

    @staticmethod
    def _format_list(items: List[Any], max_items: int = 5) -> str:
        """
        Format a list for display, truncating if too long.

        Args:
            items: List of items to format
            max_items: Maximum number of items to display

        Returns:
            Formatted string representation
        """
        if len(items) <= max_items:
            return ", ".join(str(item) for item in items)
        else:
            shown = ", ".join(str(item) for item in items[:max_items])
            return f"{shown}, ... ({len(items)} total)"


def format_results_table(result: ProfileResult) -> Table:
    """
    Convenience function to create a results table.

    Args:
        result: ProfileResult to format

    Returns:
        Rich Table object

    Example:
        >>> table = format_results_table(result)
        >>> Console().print(table)
    """
    formatter = ResultFormatter()
    return formatter.format_results_table(result)


def format_metrics(metrics: PerformanceMetrics) -> str:
    """
    Convenience function to format metrics.

    Args:
        metrics: PerformanceMetrics to format

    Returns:
        Formatted string

    Example:
        >>> formatted = format_metrics(metrics)
        >>> print(formatted)
    """
    formatter = ResultFormatter()
    return formatter.format_metrics(metrics)


def format_profile_result(result: ProfileResult) -> Panel:
    """
    Format a complete ProfileResult with summary and table.

    Args:
        result: ProfileResult to format

    Returns:
        Rich Panel containing formatted result

    Example:
        >>> panel = format_profile_result(result)
        >>> Console().print(panel)
    """
    formatter = ResultFormatter()

    # Create summary and table
    summary = formatter.format_summary_panel(result)
    table = formatter.format_results_table(result)

    # Combine into output
    from rich.console import Group
    group = Group(summary, "", table)

    return group


__all__ = [
    "ResultFormatter",
    "format_results_table",
    "format_metrics",
    "format_profile_result",
]
