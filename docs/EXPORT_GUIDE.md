# Export Functionality Guide

This document provides comprehensive information about the export functionality for the Big-O Complexity Analyzer.

## Overview

The export functionality allows you to save profiling results in multiple formats:
- **JSON**: Full detailed results with all metrics and statistics
- **CSV**: Tabular format with one row per input size for easy analysis in spreadsheets
- **Summary CSV**: Quick overview of algorithm characteristics

## Architecture

### Core Components

#### 1. **ProfileResult** (`complexity-profiler/analysis/profiler.py`)

The `ProfileResult` dataclass is the central data structure that holds all profiling results:

```python
@dataclass
class ProfileResult:
    algorithm_name: str                          # Name of profiled algorithm
    category: str                                # Algorithm category
    input_sizes: list[int]                       # Tested input sizes
    metrics_per_size: dict[int, list[PerformanceMetrics]]   # Metrics per size
    statistics_per_size: dict[int, Statistics]   # Statistics per size
    empirical_complexity: str                    # Detected complexity (e.g., "O(n log n)")
    r_squared: float                             # Fit quality (0.0-1.0)
    timestamp: datetime                          # When profiling was done
    notes: Optional[str]                         # Optional notes
```

**Key Methods:**
- `to_dict()`: Converts ProfileResult to dictionary for serialization
- `get_summary()`: Returns high-level summary information

#### 2. **JSON Exporter** (`complexity-profiler/export/json_exporter.py`)

Handles JSON serialization with support for numpy types and datetime objects.

**Features:**
- Custom `NumpyEncoder` for handling numpy types (int64, float64, ndarray, etc.)
- Pretty printing option with indentation and key sorting
- Automatic parent directory creation
- Comprehensive error handling with custom exceptions

**Functions:**
- `export_to_json(result, file_path, pretty=True)`: Export results to JSON
- `load_from_json(file_path)`: Load results from JSON file

**Example:**
```python
from pathlib import Path
from complexity-profiler.export import export_to_json

export_to_json(profile_result, Path("results.json"), pretty=True)
```

#### 3. **CSV Exporter** (`complexity-profiler/export/csv_exporter.py`)

Converts profiling results to pandas DataFrames for CSV export.

**Features:**
- One row per input size with aggregated statistics
- Comprehensive column set covering all metrics and statistics
- Support for multiple summary metrics (min, max, mean)
- Data validation before export

**Functions:**
- `export_to_csv(result, file_path)`: Export full results to CSV
- `export_metrics_summary_csv(result, file_path)`: Export summary to CSV
- `load_from_csv(file_path, algorithm_name, category)`: Load CSV data

**CSV Columns:**
```
input_size, num_runs, algorithm, category,
mean_comparisons, mean_swaps, mean_accesses,
mean_recursive_calls, mean_memory_operations,
mean_execution_time, mean_total_operations,
min_execution_time, max_execution_time,
stat_mean, stat_median, stat_std_dev,
stat_min, stat_max, stat_p25, stat_p75, stat_cv
```

## Usage Examples

### Export to JSON (Pretty Printed)

```python
from datetime import datetime
from pathlib import Path
from complexity-profiler.analysis.profiler import ProfileResult
from complexity-profiler.analysis.metrics import PerformanceMetrics
from complexity-profiler.analysis.statistics import Statistics
from complexity-profiler.export import export_to_json

# Create ProfileResult (details omitted for brevity)
result = ProfileResult(
    algorithm_name="QuickSort",
    category="sorting",
    input_sizes=[100, 1000, 10000],
    metrics_per_size={...},
    statistics_per_size={...},
    empirical_complexity="O(n log n)",
    r_squared=0.98,
)

# Export with pretty printing
export_to_json(result, Path("quicksort_results.json"), pretty=True)
```

**Output JSON (snippet):**
```json
{
  "algorithm_name": "QuickSort",
  "category": "sorting",
  "empirical_complexity": "O(n log n)",
  "input_sizes": [100, 1000, 10000],
  "metrics_per_size": {
    "100": [
      {
        "accesses": 50,
        "comparisons": 75,
        "execution_time": 0.001234,
        ...
      }
    ]
  },
  "r_squared": 0.98,
  ...
}
```

### Export to CSV

```python
from complexity-profiler.export import export_to_csv

# Export to CSV for spreadsheet analysis
export_to_csv(result, Path("quicksort_results.csv"))
```

**Output CSV (snippet):**
```csv
input_size,num_runs,algorithm,category,mean_comparisons,mean_swaps,...
100,3,QuickSort,sorting,75.0,45.0,...
1000,3,QuickSort,sorting,750.0,450.0,...
10000,3,QuickSort,sorting,7500.0,4500.0,...
```

### Export Summary

```python
from complexity-profiler.export import export_metrics_summary_csv

# Export just the summary
export_metrics_summary_csv(result, Path("quicksort_summary.csv"))
```

**Output Summary CSV:**
```csv
algorithm_name,category,input_sizes,empirical_complexity,r_squared,timestamp,notes
QuickSort,sorting,100;1000;10000,O(n log n),0.98,2024-12-07T10:30:45.123456,Sample profiling
```

### Load from JSON

```python
from complexity-profiler.export import load_from_json

# Load previously saved results
result = load_from_json(Path("quicksort_results.json"))
print(f"Algorithm: {result.algorithm_name}")
print(f"Complexity: {result.empirical_complexity}")
print(f"Fit Quality (R²): {result.r_squared}")
```

### Load from CSV

```python
from complexity-profiler.export import load_from_csv

# Load CSV data back into DataFrame
df = load_from_csv(Path("quicksort_results.csv"), "QuickSort", "sorting")
print(df[['input_size', 'mean_execution_time']])
```

## Error Handling

All export functions use custom `ExportError` exceptions for graceful error handling:

```python
from complexity-profiler.utils.exceptions import ExportError
from complexity-profiler.export import export_to_json

try:
    export_to_json(result, Path("/invalid/path/results.json"))
except ExportError as e:
    print(f"Export failed: {e.message}")
    print(f"Details: {e.details}")
```

**Error Types:**
- **Serialization Errors**: When data cannot be converted to JSON
- **I/O Errors**: When file operations fail
- **Validation Errors**: When data is invalid or incomplete
- **File Not Found**: When loading from non-existent files
- **Parse Errors**: When loading malformed CSV/JSON files

## Data Structure Details

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    comparisons: int           # Comparison operations
    swaps: int                 # Swap/exchange operations
    accesses: int              # Array/data structure accesses
    recursive_calls: int       # Recursive function calls
    execution_time: float      # Total execution time (seconds)
    memory_operations: int     # Memory allocation/copy operations
```

### Statistics

```python
@dataclass
class Statistics:
    mean: float                # Arithmetic mean
    median: float              # Middle value
    std_dev: float             # Standard deviation
    min: float                 # Minimum value
    max: float                 # Maximum value
    percentile_25: float       # Q1 (25th percentile)
    percentile_75: float       # Q3 (75th percentile)
    coefficient_of_variation: float  # Std dev / mean ratio
```

## Type Hints and Docstrings

All export functions include:
- **Complete type hints**: Full signature documentation
- **Comprehensive docstrings**: Detailed parameter and return value documentation
- **Usage examples**: Practical code examples in docstrings
- **Error documentation**: Lists all possible exceptions

Example:
```python
def export_to_json(
    result: ProfileResult,
    file_path: Path,
    pretty: bool = True,
) -> None:
    """
    Export profiling results to a JSON file.

    Args:
        result: ProfileResult object containing the profiling data
        file_path: Path where the JSON file should be written
        pretty: If True, format JSON with indentation and sorting

    Raises:
        ExportError: If the export operation fails

    Example:
        >>> from pathlib import Path
        >>> export_to_json(result, Path("results.json"))
    """
```

## Performance Considerations

### JSON Export
- Suitable for complete data preservation
- Larger file size than CSV
- Retains all numeric precision
- Human-readable with pretty printing

### CSV Export
- Optimized for tabular data analysis
- Smaller file size
- One row per input size (aggregated statistics)
- Compatible with spreadsheet applications

### Memory Usage
- Both exporters work in-memory
- Suitable for typical profiling datasets (hundreds to thousands of rows)
- For very large datasets (>100k rows), consider streaming approaches

## Integration Example

```python
from pathlib import Path
from complexity-profiler.analysis.profiler import ProfileResult
from complexity-profiler.export import export_to_json, export_to_csv

def save_profiling_results(result: ProfileResult, output_dir: str) -> None:
    """Save profiling results in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export to both formats
    json_file = output_path / f"{result.algorithm_name}_results.json"
    csv_file = output_path / f"{result.algorithm_name}_results.csv"

    export_to_json(result, json_file, pretty=True)
    export_to_csv(result, csv_file)

    print(f"Results saved to {json_file} and {csv_file}")

# Usage
save_profiling_results(result, "profiling_results")
```

## Dependencies

- **json**: Standard library for JSON serialization
- **pandas**: For DataFrame creation and CSV operations
- **pathlib**: For path handling
- **dataclasses**: For dataclass conversion utilities
- **numpy**: For numpy type handling in JSON encoder

## Future Enhancements

Potential additions to the export functionality:
- HTML report generation with visualizations
- Excel export with multiple sheets and formatting
- SQL database export
- Streaming export for large datasets
- Compression support (gzip, bzip2)
- Data validation schemas
- Version control for results
