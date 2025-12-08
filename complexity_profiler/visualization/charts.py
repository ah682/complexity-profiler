"""
Chart generation for visualizing algorithm complexity and performance.

This module provides comprehensive visualization capabilities for profiling results,
including complexity curves, comparison plots, and statistical representations.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import seaborn as sns

from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.visualization.themes import (
    CHART_STYLE,
    COLORS,
    METRIC_MARKERS,
    LINE_STYLES,
    EXPORT_FORMATS,
    get_color_for_algorithm,
    get_color_for_complexity,
)


class ChartGenerator:
    """
    Generator for creating professional complexity analysis visualizations.

    This class provides methods to create various types of charts for analyzing
    and comparing algorithm performance, including complexity curves, scatter plots,
    and multi-algorithm comparisons.

    Example:
        >>> generator = ChartGenerator()
        >>> generator.plot_complexity_curve(result, metric="execution_time")
        >>> generator.save("output.png")
    """

    def __init__(self, style: Optional[dict] = None) -> None:
        """
        Initialize the chart generator with optional custom styling.

        Args:
            style: Optional dictionary of matplotlib style parameters.
                  If None, uses default CHART_STYLE from themes module.
        """
        self.style = style or CHART_STYLE
        self._apply_style()
        self.current_figure: Optional[mpl_figure.Figure] = None

    def _apply_style(self) -> None:
        """Apply the configured style to matplotlib."""
        plt.rcParams.update(self.style)
        sns.set_palette(COLORS)

    def plot_complexity_curve(
        self,
        result: ProfileResult,
        metric: str = "execution_time",
        save_path: Optional[Path] = None,
        show_error_bars: bool = True,
        log_scale: bool = False,
    ) -> mpl_figure.Figure:
        """
        Plot complexity curve for a single algorithm with fitted line.

        Creates a professional chart showing:
        - Measured data points with error bars (optional)
        - Mean trend line
        - Detected complexity class annotation
        - Log-log scale option for better visualization

        Args:
            result: ProfileResult containing the data to plot
            metric: Metric to plot ("execution_time", "comparisons", "swaps", etc.)
            save_path: Optional path to save the figure
            show_error_bars: Whether to display error bars (default: True)
            log_scale: Whether to use log-log scale (default: False)

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If metric is invalid

        Example:
            >>> fig = generator.plot_complexity_curve(
            ...     result,
            ...     metric="execution_time",
            ...     log_scale=True
            ... )
        """
        # Extract data based on metric
        sizes = np.array(result.input_sizes)
        mean_values, std_values = self._extract_metric_data(result, metric)

        # Create figure
        fig, ax = plt.subplots(figsize=self.style.get("figure.figsize", (12, 7)))
        self.current_figure = fig

        # Plot data points with error bars
        color = get_color_for_complexity(result.empirical_complexity)
        marker = METRIC_MARKERS.get(metric, "o")

        if show_error_bars and std_values is not None:
            ax.errorbar(
                sizes,
                mean_values,
                yerr=std_values,
                fmt=marker,
                color=color,
                markersize=8,
                capsize=5,
                capthick=2,
                label="Measured",
                alpha=0.7,
                elinewidth=2,
            )
        else:
            ax.plot(
                sizes,
                mean_values,
                marker=marker,
                color=color,
                linestyle=LINE_STYLES["measured"],
                linewidth=2.5,
                markersize=8,
                label="Measured",
                alpha=0.8,
            )

        # Add fitted curve line
        ax.plot(
            sizes,
            mean_values,
            linestyle=LINE_STYLES["fitted"],
            color=color,
            linewidth=3,
            alpha=0.5,
            label=f"Fitted {result.empirical_complexity}",
        )

        # Configure axes
        ax.set_xlabel("Input Size (n)", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            self._format_metric_label(metric),
            fontsize=12,
            fontweight="bold"
        )
        ax.set_title(
            f"{result.algorithm_name} - Complexity Analysis",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Apply log scale if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"{result.algorithm_name} - Complexity Analysis (Log-Log Scale)",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

        # Add complexity and R² annotation
        textstr = f"Complexity: {result.empirical_complexity}\nR² = {result.r_squared:.4f}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

        # Configure legend
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Tight layout
        fig.tight_layout()

        # Save if path provided
        if save_path:
            self.save(save_path)

        return fig

    def plot_comparison(
        self,
        results: List[ProfileResult],
        metric: str = "execution_time",
        save_path: Optional[Path] = None,
        log_scale: bool = False,
    ) -> mpl_figure.Figure:
        """
        Plot comparison of multiple algorithms on the same chart.

        Creates a multi-line chart comparing performance of different algorithms
        across the same input sizes.

        Args:
            results: List of ProfileResult objects to compare
            metric: Metric to plot (default: "execution_time")
            save_path: Optional path to save the figure
            log_scale: Whether to use log-log scale (default: False)

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If results list is empty or metric is invalid

        Example:
            >>> results = [merge_result, quick_result, bubble_result]
            >>> fig = generator.plot_comparison(results, log_scale=True)
        """
        if not results:
            raise ValueError("results list cannot be empty")

        # Create figure
        fig, ax = plt.subplots(figsize=self.style.get("figure.figsize", (12, 7)))
        self.current_figure = fig

        # Plot each algorithm
        for idx, result in enumerate(results):
            sizes = np.array(result.input_sizes)
            mean_values, _ = self._extract_metric_data(result, metric)

            color = get_color_for_algorithm(idx)
            marker = METRIC_MARKERS.get(metric, "o")

            ax.plot(
                sizes,
                mean_values,
                marker=marker,
                color=color,
                linestyle=LINE_STYLES["measured"],
                linewidth=2.5,
                markersize=8,
                label=f"{result.algorithm_name} ({result.empirical_complexity})",
                alpha=0.8,
            )

        # Configure axes
        ax.set_xlabel("Input Size (n)", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            self._format_metric_label(metric),
            fontsize=12,
            fontweight="bold"
        )
        ax.set_title(
            f"Algorithm Comparison - {self._format_metric_label(metric)}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Apply log scale if requested
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"Algorithm Comparison - {self._format_metric_label(metric)} (Log-Log)",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

        # Configure legend
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Tight layout
        fig.tight_layout()

        # Save if path provided
        if save_path:
            self.save(save_path)

        return fig

    def plot_metrics_overview(
        self,
        result: ProfileResult,
        save_path: Optional[Path] = None,
    ) -> mpl_figure.Figure:
        """
        Plot overview of multiple metrics for a single algorithm.

        Creates a 2x2 subplot grid showing execution time, comparisons,
        swaps, and accesses.

        Args:
            result: ProfileResult containing the data to plot
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object

        Example:
            >>> fig = generator.plot_metrics_overview(result)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.current_figure = fig

        metrics = [
            ("execution_time", "Execution Time"),
            ("comparisons", "Comparisons"),
            ("swaps", "Swaps"),
            ("accesses", "Array Accesses"),
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            sizes = np.array(result.input_sizes)
            mean_values, std_values = self._extract_metric_data(result, metric)

            color = get_color_for_algorithm(idx)
            marker = METRIC_MARKERS.get(metric, "o")

            # Plot with error bars
            if std_values is not None:
                ax.errorbar(
                    sizes,
                    mean_values,
                    yerr=std_values,
                    fmt=marker,
                    color=color,
                    markersize=6,
                    capsize=4,
                    label=title,
                    alpha=0.7,
                )
            else:
                ax.plot(
                    sizes,
                    mean_values,
                    marker=marker,
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=title,
                    alpha=0.8,
                )

            ax.set_xlabel("Input Size", fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{result.algorithm_name} - Metrics Overview",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        fig.tight_layout()

        if save_path:
            self.save(save_path)

        return fig

    def save(self, path: Path, dpi: int = 300, format: Optional[str] = None) -> None:
        """
        Save the current figure to a file.

        Args:
            path: Path to save the figure
            dpi: Dots per inch for raster formats (default: 300)
            format: File format (png, svg, pdf). If None, inferred from path.

        Raises:
            ValueError: If no figure has been created yet
            IOError: If save fails

        Example:
            >>> generator.plot_complexity_curve(result)
            >>> generator.save(Path("output.png"), dpi=300)
        """
        if self.current_figure is None:
            raise ValueError("No figure to save. Create a plot first.")

        # Determine format
        if format is None:
            format = path.suffix.lstrip(".")

        # Get export settings
        export_settings = EXPORT_FORMATS.get(format, EXPORT_FORMATS["png"]).copy()
        export_settings["dpi"] = dpi

        # Save figure
        self.current_figure.savefig(str(path), **export_settings)

    def close(self) -> None:
        """Close the current figure and free resources."""
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None

    def _extract_metric_data(
        self,
        result: ProfileResult,
        metric: str
    ) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """
        Extract mean and standard deviation for a metric from ProfileResult.

        Args:
            result: ProfileResult to extract from
            metric: Metric name

        Returns:
            Tuple of (mean_values, std_values) as numpy arrays

        Raises:
            ValueError: If metric is invalid
        """
        valid_metrics = {
            "execution_time",
            "comparisons",
            "swaps",
            "accesses",
            "total_operations",
        }

        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
            )

        mean_values = []
        std_values = []

        for size in result.input_sizes:
            stats = result.statistics_per_size[size]

            # For execution_time, use statistics directly
            if metric == "execution_time":
                mean_values.append(stats.mean)
                std_values.append(stats.std_dev)
            else:
                # For other metrics, extract from metrics_per_size
                metrics_list = result.metrics_per_size[size]
                values = [getattr(m, metric) for m in metrics_list]

                if metric == "total_operations":
                    values = [m.total_operations() for m in metrics_list]

                mean_values.append(np.mean(values))
                std_values.append(np.std(values))

        return (
            np.array(mean_values, dtype=np.float64),
            np.array(std_values, dtype=np.float64),
        )

    @staticmethod
    def _format_metric_label(metric: str) -> str:
        """
        Format metric name for display labels.

        Args:
            metric: Metric identifier

        Returns:
            Formatted label string

        Example:
            >>> ChartGenerator._format_metric_label("execution_time")
            'Execution Time (seconds)'
        """
        labels = {
            "execution_time": "Execution Time (seconds)",
            "comparisons": "Number of Comparisons",
            "swaps": "Number of Swaps",
            "accesses": "Number of Array Accesses",
            "total_operations": "Total Operations",
            "recursive_calls": "Recursive Calls",
        }
        return labels.get(metric, metric.replace("_", " ").title())


def plot_complexity_curve(
    result: ProfileResult,
    metric: str = "execution_time",
    save_path: Optional[Path] = None,
) -> mpl_figure.Figure:
    """
    Convenience function to plot a complexity curve.

    Args:
        result: ProfileResult to plot
        metric: Metric to visualize (default: "execution_time")
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_complexity_curve(result, metric="comparisons")
    """
    generator = ChartGenerator()
    return generator.plot_complexity_curve(result, metric, save_path)


def plot_comparison(
    results: List[ProfileResult],
    metric: str = "execution_time",
    save_path: Optional[Path] = None,
) -> mpl_figure.Figure:
    """
    Convenience function to plot algorithm comparison.

    Args:
        results: List of ProfileResult objects to compare
        metric: Metric to visualize (default: "execution_time")
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_comparison([result1, result2, result3])
    """
    generator = ChartGenerator()
    return generator.plot_comparison(results, metric, save_path)


__all__ = [
    "ChartGenerator",
    "plot_complexity_curve",
    "plot_comparison",
]
