"""
Input validation utilities for CLI commands.

This module provides validators for command-line arguments and options,
ensuring that user input is valid before processing.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import click

# Available algorithms registry
AVAILABLE_ALGORITHMS = {
    # Sorting algorithms
    "merge_sort": ("complexity_profiler.algorithms.sorting", "MergeSort"),
    "quick_sort": ("complexity_profiler.algorithms.sorting", "QuickSort"),
    "selection_sort": ("complexity_profiler.algorithms.sorting", "SelectionSort"),
    "bubble_sort": ("complexity_profiler.algorithms.sorting", "BubbleSort"),
    "insertion_sort": ("complexity_profiler.algorithms.sorting", "InsertionSort"),
    "heap_sort": ("complexity_profiler.algorithms.sorting", "HeapSort"),
    # Searching algorithms (if implemented)
    # "binary_search": ("complexity_profiler.algorithms.searching", "BinarySearch"),
    # "linear_search": ("complexity_profiler.algorithms.searching", "LinearSearch"),
}

# Valid metrics for visualization
VALID_METRICS = {
    "execution_time",
    "comparisons",
    "swaps",
    "accesses",
    "total_operations",
    "recursive_calls",
}

# Valid data types for test data generation
VALID_DATA_TYPES = {
    "random",      # Random unsorted data
    "sorted",      # Already sorted data (best case)
    "reversed",    # Reverse sorted data (worst case)
    "nearly_sorted",  # Mostly sorted with some disorder
}

# Valid export formats
VALID_EXPORT_FORMATS = {
    "json",
    "csv",
    "html",
}


def validate_algorithm_name(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validate algorithm name parameter.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Algorithm name to validate

    Returns:
        Validated algorithm name

    Raises:
        click.BadParameter: If algorithm name is invalid

    Example:
        >>> validate_algorithm_name(ctx, param, "merge_sort")
        'merge_sort'
    """
    if value not in AVAILABLE_ALGORITHMS:
        available = ", ".join(sorted(AVAILABLE_ALGORITHMS.keys()))
        raise click.BadParameter(
            f"Unknown algorithm '{value}'. Available algorithms: {available}"
        )
    return value


def validate_algorithm_names(
    ctx: click.Context,
    param: click.Parameter,
    value: Tuple[str, ...]
) -> Tuple[str, ...]:
    """
    Validate multiple algorithm names.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Tuple of algorithm names

    Returns:
        Validated tuple of algorithm names

    Raises:
        click.BadParameter: If any algorithm name is invalid
    """
    for algo_name in value:
        validate_algorithm_name(ctx, param, algo_name)
    return value


def validate_input_sizes(ctx: click.Context, param: click.Parameter, value: str) -> List[int]:
    """
    Validate and parse input sizes parameter.

    Accepts formats like:
    - "100,500,1000" - comma-separated list
    - "100-1000:5" - range with step count
    - "100" - single size

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Input sizes string to parse

    Returns:
        List of validated input sizes

    Raises:
        click.BadParameter: If format is invalid or values are not positive

    Example:
        >>> validate_input_sizes(ctx, param, "100,500,1000")
        [100, 500, 1000]
        >>> validate_input_sizes(ctx, param, "100-1000:5")
        [100, 325, 550, 775, 1000]
    """
    if not value:
        raise click.BadParameter("Input sizes cannot be empty")

    try:
        # Check for range notation (e.g., "100-1000:5")
        if "-" in value and ":" in value:
            range_part, step_count = value.split(":")
            start, end = range_part.split("-")
            start_val = int(start)
            end_val = int(end)
            steps = int(step_count)

            if start_val <= 0 or end_val <= 0:
                raise click.BadParameter("Input sizes must be positive integers")
            if start_val >= end_val:
                raise click.BadParameter("Start size must be less than end size")
            if steps < 2:
                raise click.BadParameter("Step count must be at least 2")

            # Generate evenly spaced sizes
            import numpy as np
            sizes = np.linspace(start_val, end_val, steps, dtype=int).tolist()
            return sorted(list(set(sizes)))  # Remove duplicates and sort

        # Check for comma-separated list
        elif "," in value:
            sizes = [int(s.strip()) for s in value.split(",")]
            if any(s <= 0 for s in sizes):
                raise click.BadParameter("All input sizes must be positive integers")
            return sorted(sizes)

        # Single value
        else:
            size = int(value)
            if size <= 0:
                raise click.BadParameter("Input size must be a positive integer")
            return [size]

    except ValueError as e:
        raise click.BadParameter(
            f"Invalid input sizes format: {value}. "
            "Use comma-separated values (e.g., '100,500,1000') "
            "or range notation (e.g., '100-1000:5')"
        )


def validate_runs(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """
    Validate runs per size parameter.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Number of runs

    Returns:
        Validated number of runs

    Raises:
        click.BadParameter: If value is not positive or too large
    """
    if value < 1:
        raise click.BadParameter("Number of runs must be at least 1")
    if value > 100:
        raise click.BadParameter(
            "Number of runs is too large (max 100). "
            "Large values may significantly increase execution time."
        )
    return value


def validate_data_type(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validate data type parameter.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Data type to validate

    Returns:
        Validated data type

    Raises:
        click.BadParameter: If data type is invalid
    """
    if value not in VALID_DATA_TYPES:
        valid_types = ", ".join(sorted(VALID_DATA_TYPES))
        raise click.BadParameter(
            f"Invalid data type '{value}'. Valid types: {valid_types}"
        )
    return value


def validate_export_format(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validate export format parameter.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Export format to validate

    Returns:
        Validated export format

    Raises:
        click.BadParameter: If format is invalid
    """
    if value not in VALID_EXPORT_FORMATS:
        valid_formats = ", ".join(sorted(VALID_EXPORT_FORMATS))
        raise click.BadParameter(
            f"Invalid export format '{value}'. Valid formats: {valid_formats}"
        )
    return value


def validate_output_path(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[Path]:
    """
    Validate output path parameter.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Output path string

    Returns:
        Validated Path object or None

    Raises:
        click.BadParameter: If path is invalid or directory doesn't exist
    """
    if value is None:
        return None

    path = Path(value)

    # Check if parent directory exists
    if not path.parent.exists():
        raise click.BadParameter(
            f"Parent directory does not exist: {path.parent}"
        )

    # Warn if file already exists
    if path.exists():
        if not click.confirm(f"File {path} already exists. Overwrite?", default=False):
            raise click.Abort()

    return path


def validate_metric(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validate metric parameter for visualization.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Metric name to validate

    Returns:
        Validated metric name

    Raises:
        click.BadParameter: If metric is invalid
    """
    if value not in VALID_METRICS:
        valid_metrics = ", ".join(sorted(VALID_METRICS))
        raise click.BadParameter(
            f"Invalid metric '{value}'. Valid metrics: {valid_metrics}"
        )
    return value


def get_algorithm_instance(algorithm_name: str):
    """
    Get an instance of the specified algorithm.

    Args:
        algorithm_name: Name of the algorithm (e.g., "merge_sort")

    Returns:
        Instance of the algorithm class

    Raises:
        ValueError: If algorithm cannot be loaded

    Example:
        >>> algo = get_algorithm_instance("merge_sort")
        >>> print(algo.metadata.name)
        Merge Sort
    """
    if algorithm_name not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    module_path, class_name = AVAILABLE_ALGORITHMS[algorithm_name]

    try:
        # Dynamic import
        import importlib
        module = importlib.import_module(module_path)
        algorithm_class = getattr(module, class_name)
        return algorithm_class()
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Failed to load algorithm '{algorithm_name}': {e}"
        )


def create_data_generator(data_type: str, seed: Optional[int] = None):
    """
    Create a data generator function based on data type.

    Args:
        data_type: Type of data to generate
        seed: Optional random seed for reproducibility

    Returns:
        Function that takes size and returns list of data

    Example:
        >>> generator = create_data_generator("random", seed=42)
        >>> data = generator(100)
        >>> len(data)
        100
    """
    import random
    import numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if data_type == "random":
        return lambda n: list(np.random.randint(0, 10000, size=n))

    elif data_type == "sorted":
        return lambda n: list(range(n))

    elif data_type == "reversed":
        return lambda n: list(range(n, 0, -1))

    elif data_type == "nearly_sorted":
        def nearly_sorted(n):
            data = list(range(n))
            # Swap ~5% of elements
            num_swaps = max(1, n // 20)
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                data[i], data[j] = data[j], data[i]
            return data
        return nearly_sorted

    else:
        raise ValueError(f"Invalid data type: {data_type}")


__all__ = [
    "AVAILABLE_ALGORITHMS",
    "VALID_METRICS",
    "VALID_DATA_TYPES",
    "VALID_EXPORT_FORMATS",
    "validate_algorithm_name",
    "validate_algorithm_names",
    "validate_input_sizes",
    "validate_runs",
    "validate_data_type",
    "validate_export_format",
    "validate_output_path",
    "validate_metric",
    "get_algorithm_instance",
    "create_data_generator",
]
