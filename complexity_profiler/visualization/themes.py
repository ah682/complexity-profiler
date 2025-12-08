"""
Visual themes and styling configurations for charts and displays.

This module defines color schemes, chart styles, and other visual properties
used throughout the visualization components to ensure consistent, professional
appearance.
"""

from typing import Dict, List, Any


# Color palette for charts - carefully selected for visual distinction and accessibility
COLORS: List[str] = [
    "#2E86AB",  # Blue - primary color for single charts
    "#A23B72",  # Magenta - secondary color
    "#F18F01",  # Orange - tertiary color
    "#C73E1D",  # Red - quaternary color
    "#6A994E",  # Green - quinary color
    "#BC4B51",  # Burgundy
    "#8B5A3C",  # Brown
    "#5F4B8B",  # Purple
    "#2F4858",  # Dark blue
    "#E63946",  # Bright red
]


# Matplotlib/Seaborn chart styling configuration
CHART_STYLE: Dict[str, Any] = {
    # Figure settings
    "figure.figsize": (12, 7),
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "figure.edgecolor": "white",

    # Axes settings
    "axes.facecolor": "#F8F9FA",
    "axes.edgecolor": "#DEE2E6",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "axes.labelcolor": "#212529",
    "axes.titlecolor": "#212529",

    # Grid settings
    "grid.color": "#CED4DA",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,

    # Line settings
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "lines.markeredgewidth": 1.5,

    # Legend settings
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.facecolor": "white",
    "legend.edgecolor": "#DEE2E6",
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,

    # Tick settings
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#495057",
    "ytick.color": "#495057",
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 10,
}


# Rich console color scheme for terminal output
CONSOLE_THEME: Dict[str, str] = {
    "header": "bold cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "blue",
    "metric_label": "bold white",
    "metric_value": "cyan",
    "complexity_class": "magenta",
    "algorithm_name": "bold green",
    "table_header": "bold cyan",
    "path": "yellow",
}


# Complexity class color mapping for charts
COMPLEXITY_COLORS: Dict[str, str] = {
    "O(1)": "#2E86AB",          # Blue - constant
    "O(log n)": "#6A994E",      # Green - logarithmic
    "O(n)": "#F18F01",          # Orange - linear
    "O(n log n)": "#A23B72",    # Magenta - linearithmic
    "O(n²)": "#C73E1D",         # Red - quadratic
    "O(n³)": "#BC4B51",         # Burgundy - cubic
    "O(2^n)": "#8B5A3C",        # Brown - exponential
    "O(n!)": "#E63946",         # Bright red - factorial
}


# Marker styles for different metrics in charts
METRIC_MARKERS: Dict[str, str] = {
    "execution_time": "o",       # Circle
    "comparisons": "s",          # Square
    "swaps": "^",                # Triangle up
    "accesses": "D",             # Diamond
    "recursive_calls": "v",      # Triangle down
    "total_operations": "p",     # Pentagon
    "memory_operations": "h",    # Hexagon
}


# Line styles for curve fitting and theoretical complexity
LINE_STYLES: Dict[str, str] = {
    "measured": "-",          # Solid line for measured data
    "fitted": "--",           # Dashed line for fitted curves
    "theoretical": "-.",      # Dash-dot for theoretical complexity
    "confidence": ":",        # Dotted for confidence intervals
}


# Export format configurations
EXPORT_FORMATS: Dict[str, Dict[str, Any]] = {
    "png": {
        "dpi": 300,
        "format": "png",
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
    },
    "svg": {
        "format": "svg",
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
    },
    "pdf": {
        "format": "pdf",
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
    },
}


def get_color_for_algorithm(index: int) -> str:
    """
    Get a distinct color for an algorithm based on its index.

    Uses modulo to cycle through colors if there are more algorithms
    than available colors.

    Args:
        index: Zero-based index of the algorithm

    Returns:
        Hex color code string

    Example:
        >>> get_color_for_algorithm(0)
        '#2E86AB'
        >>> get_color_for_algorithm(10)
        '#2E86AB'  # Cycles back to first color
    """
    return COLORS[index % len(COLORS)]


def get_color_for_complexity(complexity: str) -> str:
    """
    Get the designated color for a complexity class.

    Args:
        complexity: Complexity class string (e.g., "O(n log n)")

    Returns:
        Hex color code string, defaults to blue if complexity not found

    Example:
        >>> get_color_for_complexity("O(n log n)")
        '#A23B72'
        >>> get_color_for_complexity("O(unknown)")
        '#2E86AB'  # Default blue
    """
    return COMPLEXITY_COLORS.get(complexity, COLORS[0])


__all__ = [
    "COLORS",
    "CHART_STYLE",
    "CONSOLE_THEME",
    "COMPLEXITY_COLORS",
    "METRIC_MARKERS",
    "LINE_STYLES",
    "EXPORT_FORMATS",
    "get_color_for_algorithm",
    "get_color_for_complexity",
]
