"""
CSV export functionality for profiling results.

This module provides utilities for exporting algorithm profiling results
to CSV format using pandas DataFrames. Each input size gets one row with
comprehensive statistics.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.utils.exceptions import ExportError


def export_to_csv(
    result: ProfileResult,
    file_path: Path,
) -> None:
    """
    Export profiling results to a CSV file.

    Converts a ProfileResult object to a pandas DataFrame and exports it
    to CSV format. Each input size is represented as a single row with
    aggregated statistics for all metrics collected at that size.

    The CSV includes columns for:
    - input_size: The input size used for profiling
    - metrics (comparisons, swaps, accesses, etc.): Mean values across runs
    - statistics (mean, median, std_dev, min, max, percentiles): Computed metrics

    Args:
        result: ProfileResult object containing the profiling data
        file_path: Path where the CSV file should be written

    Raises:
        ExportError: If the export operation fails due to I/O errors,
                    invalid file paths, or data conversion issues

    Example:
        >>> from complexity_profiler.analysis.profiler import ProfileResult
        >>> from pathlib import Path
        >>> result = ProfileResult(...)
        >>> export_to_csv(result, Path("results.csv"))
    """
    try:
        # Ensure parent directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Build data for DataFrame
        rows = []
        for input_size in result.input_sizes:
            metrics_list = result.metrics_per_size.get(input_size, [])
            statistics = result.statistics_per_size.get(input_size)

            if not metrics_list or statistics is None:
                continue

            # Calculate mean metrics across all runs for this input size
            row = {
                'input_size': input_size,
                'num_runs': len(metrics_list),
                'algorithm': result.algorithm_name,
                'category': result.category,
            }

            # Add mean metrics
            if metrics_list:
                row['mean_comparisons'] = sum(m.comparisons for m in metrics_list) / len(metrics_list)
                row['mean_swaps'] = sum(m.swaps for m in metrics_list) / len(metrics_list)
                row['mean_accesses'] = sum(m.accesses for m in metrics_list) / len(metrics_list)
                row['mean_recursive_calls'] = sum(m.recursive_calls for m in metrics_list) / len(metrics_list)
                row['mean_memory_operations'] = sum(m.memory_operations for m in metrics_list) / len(metrics_list)
                row['mean_execution_time'] = sum(m.execution_time for m in metrics_list) / len(metrics_list)
                row['mean_total_operations'] = sum(m.total_operations() for m in metrics_list) / len(metrics_list)

                # Add min/max for execution time
                row['min_execution_time'] = min(m.execution_time for m in metrics_list)
                row['max_execution_time'] = max(m.execution_time for m in metrics_list)

            # Add statistics
            if statistics:
                row['stat_mean'] = statistics.mean
                row['stat_median'] = statistics.median
                row['stat_std_dev'] = statistics.std_dev
                row['stat_min'] = statistics.min
                row['stat_max'] = statistics.max
                row['stat_p25'] = statistics.percentile_25
                row['stat_p75'] = statistics.percentile_75
                row['stat_cv'] = statistics.coefficient_of_variation

            rows.append(row)

        if not rows:
            raise ValueError("No data available to export")

        # Create DataFrame and write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, encoding='utf-8')

    except ValueError as e:
        raise ExportError(
            f"Invalid data for CSV export: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": "invalid_data"},
        ) from e
    except (IOError, OSError) as e:
        raise ExportError(
            f"Failed to write CSV file: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": "io_error"},
        ) from e
    except Exception as e:
        raise ExportError(
            f"Unexpected error during CSV export: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": type(e).__name__},
        ) from e


def export_metrics_summary_csv(
    result: ProfileResult,
    file_path: Path,
) -> None:
    """
    Export a summary of profiling results to CSV.

    Creates a simpler CSV file with just the summary information:
    algorithm name, category, empirical complexity, R-squared fit, etc.
    This is useful for quick overview comparisons across multiple algorithms.

    Args:
        result: ProfileResult object containing the profiling data
        file_path: Path where the summary CSV file should be written

    Raises:
        ExportError: If the export operation fails

    Example:
        >>> from pathlib import Path
        >>> export_metrics_summary_csv(result, Path("summary.csv"))
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary row
        summary = result.get_summary()
        summary_data = {
            'algorithm_name': summary['algorithm_name'],
            'category': summary['category'],
            'input_sizes': ','.join(map(str, summary['input_sizes'])),
            'empirical_complexity': summary['empirical_complexity'],
            'r_squared': summary['r_squared'],
            'timestamp': summary['timestamp'],
            'notes': summary['notes'],
        }

        # Convert to DataFrame and write
        df = pd.DataFrame([summary_data])
        df.to_csv(file_path, index=False, encoding='utf-8')

    except (IOError, OSError) as e:
        raise ExportError(
            f"Failed to write summary CSV file: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": "io_error"},
        ) from e
    except Exception as e:
        raise ExportError(
            f"Unexpected error during summary CSV export: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": type(e).__name__},
        ) from e


def load_from_csv(
    file_path: Path,
    algorithm_name: str,
    category: str,
) -> pd.DataFrame:
    """
    Load profiling data from a CSV file.

    Reads a CSV file created by export_to_csv and returns it as a
    pandas DataFrame for further analysis or manipulation.

    Args:
        file_path: Path to the CSV file to load
        algorithm_name: Name of the algorithm (for validation)
        category: Category of the algorithm (for validation)

    Returns:
        Pandas DataFrame containing the profiling data

    Raises:
        ExportError: If the file cannot be read or is invalid

    Example:
        >>> from pathlib import Path
        >>> df = load_from_csv(Path("results.csv"), "quicksort", "sorting")
    """
    try:
        file_path = Path(file_path)

        # Read CSV file
        df = pd.read_csv(file_path, encoding='utf-8')

        # Validate that data matches expected algorithm
        if not df.empty:
            if (df['algorithm'].iloc[0] != algorithm_name or
                df['category'].iloc[0] != category):
                raise ValueError(
                    f"CSV data does not match expected algorithm: "
                    f"{algorithm_name} ({category})"
                )

        return df

    except FileNotFoundError as e:
        raise ExportError(
            f"CSV file not found: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": "file_not_found"},
        ) from e
    except pd.errors.ParserError as e:
        raise ExportError(
            f"Failed to parse CSV file: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": "parse_error"},
        ) from e
    except ValueError as e:
        raise ExportError(
            f"Invalid CSV data: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": "invalid_data"},
        ) from e
    except Exception as e:
        raise ExportError(
            f"Unexpected error during CSV load: {str(e)}",
            export_format="csv",
            file_path=str(file_path),
            details={"error_type": type(e).__name__},
        ) from e


__all__ = [
    "export_to_csv",
    "export_metrics_summary_csv",
    "load_from_csv",
]
