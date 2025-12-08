# Export Functionality - Quick Reference

## Quick Start

```python
from pathlib import Path
from complexity-profiler import export_to_json, export_to_csv

# Export profiling results
export_to_json(result, Path("results.json"), pretty=True)
export_to_csv(result, Path("results.csv"))
```

## Import Options

```python
# Option 1: From main package
from complexity-profiler import export_to_json, export_to_csv

# Option 2: From export subpackage
from complexity-profiler.export import export_to_json, export_to_csv

# Option 3: Import everything
from complexity-profiler.export import *
```

## JSON Export

### Basic Export
```python
export_to_json(result, Path("results.json"))
```

### Pretty Printed (Formatted)
```python
export_to_json(result, Path("results.json"), pretty=True)
```

### Compact (Minified)
```python
export_to_json(result, Path("results.json"), pretty=False)
```

### Load JSON
```python
result = load_from_json(Path("results.json"))
```

## CSV Export

### Full Export (All Data)
```python
export_to_csv(result, Path("results.csv"))
# One row per input size with all metrics and statistics
```

### Summary Only
```python
export_metrics_summary_csv(result, Path("summary.csv"))
# Single row with algorithm summary
```

### Load CSV
```python
df = load_from_csv(Path("results.csv"), "QuickSort", "sorting")
# Returns pandas DataFrame
```

## Error Handling

```python
from complexity-profiler.utils.exceptions import ExportError

try:
    export_to_json(result, Path("results.json"))
except ExportError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

## ProfileResult Structure

```python
ProfileResult(
    algorithm_name="QuickSort",          # str
    category="sorting",                  # str
    input_sizes=[100, 1000, 10000],     # list[int]
    metrics_per_size={...},              # dict[int, list[PerformanceMetrics]]
    statistics_per_size={...},           # dict[int, Statistics]
    empirical_complexity="O(n log n)",  # str
    r_squared=0.98,                      # float (0.0-1.0)
    timestamp=datetime.now(),            # datetime
    notes="Optional notes",              # Optional[str]
)
```

## Working with Results

```python
# Get summary
summary = result.get_summary()

# Convert to dict
data_dict = result.to_dict()

# Access metrics
metrics = result.metrics_per_size[1000]  # List of PerformanceMetrics

# Access statistics
stats = result.statistics_per_size[1000]  # Statistics object

# Check consistency
is_consistent = stats.is_consistent(threshold=0.15)
```

## CSV Column Reference

| Column | Description |
|--------|-------------|
| `input_size` | Size of input tested |
| `num_runs` | Number of runs performed |
| `algorithm` | Algorithm name |
| `category` | Algorithm category |
| `mean_*` | Mean values across runs |
| `min_execution_time` | Minimum execution time |
| `max_execution_time` | Maximum execution time |
| `stat_*` | Statistical measures |
| `stat_cv` | Coefficient of variation |

## JSON Structure

```json
{
  "algorithm_name": "QuickSort",
  "category": "sorting",
  "input_sizes": [100, 1000],
  "metrics_per_size": {
    "100": [
      {
        "comparisons": 75,
        "swaps": 45,
        "accesses": 52,
        "recursive_calls": 5,
        "execution_time": 0.001234,
        "memory_operations": 10
      }
    ]
  },
  "statistics_per_size": {
    "100": {
      "mean": 0.001234,
      "median": 0.001230,
      "std_dev": 0.000050,
      "min": 0.001200,
      "max": 0.001280,
      "percentile_25": 0.001220,
      "percentile_75": 0.001250,
      "coefficient_of_variation": 0.0405
    }
  },
  "empirical_complexity": "O(n log n)",
  "r_squared": 0.98,
  "timestamp": "2024-12-07T10:30:45.123456",
  "notes": "Optional profiling notes"
}
```

## Statistics Methods

```python
stats = result.statistics_per_size[1000]

# Check if measurements are consistent
is_consistent = stats.is_consistent()  # Default threshold: 0.15
is_consistent = stats.is_consistent(threshold=0.10)

# Access individual statistics
print(stats.mean)                 # Arithmetic mean
print(stats.median)               # Median
print(stats.std_dev)              # Standard deviation
print(stats.min)                  # Minimum
print(stats.max)                  # Maximum
print(stats.percentile_25)        # Q1
print(stats.percentile_75)        # Q3
print(stats.coefficient_of_variation)  # CV
```

## Common Patterns

### Export and Load Round-Trip

```python
from pathlib import Path
from complexity-profiler import export_to_json, load_from_json

# Export
export_to_json(result, Path("results.json"))

# Load back
loaded_result = load_from_json(Path("results.json"))
assert loaded_result.algorithm_name == result.algorithm_name
```

### Compare Multiple Algorithms

```python
from complexity-profiler import export_metrics_summary_csv, load_from_csv
import pandas as pd

# Export summaries for multiple algorithms
export_metrics_summary_csv(quicksort_result, Path("quicksort_summary.csv"))
export_metrics_summary_csv(mergesort_result, Path("mergesort_summary.csv"))

# Load and combine
df1 = load_from_csv(Path("quicksort_summary.csv"), "QuickSort", "sorting")
df2 = load_from_csv(Path("mergesort_summary.csv"), "MergeSort", "sorting")

# Compare
comparison = pd.concat([df1, df2])
print(comparison[['algorithm_name', 'empirical_complexity', 'r_squared']])
```

### Analyze Metrics Over Time

```python
# Get metrics for specific input size
metrics_list = result.metrics_per_size[10000]  # All runs at size 10000

# Analyze
comparisons = [m.comparisons for m in metrics_list]
execution_times = [m.execution_time for m in metrics_list]

avg_comparisons = sum(comparisons) / len(comparisons)
avg_time = sum(execution_times) / len(execution_times)

print(f"Average comparisons: {avg_comparisons}")
print(f"Average time: {avg_time:.6f}s")
```

### Generate Report

```python
def generate_report(result, output_file):
    """Generate a simple text report."""
    with open(output_file, 'w') as f:
        summary = result.get_summary()

        f.write("ALGORITHM PROFILING REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Algorithm: {summary['algorithm_name']}\n")
        f.write(f"Category: {summary['category']}\n")
        f.write(f"Complexity: {summary['empirical_complexity']}\n")
        f.write(f"R² (Fit Quality): {summary['r_squared']:.4f}\n")
        f.write(f"Timestamp: {summary['timestamp']}\n\n")

        f.write("Input Sizes Tested:\n")
        for size in summary['input_sizes']:
            stats = result.statistics_per_size[size]
            f.write(f"  {size}: mean={stats.mean:.6f}s, cv={stats.coefficient_of_variation:.4f}\n")

# Usage
generate_report(result, "report.txt")
```

## Error Messages

```python
# File not found
ExportError("JSON file not found", export_format="json", file_path="/invalid/path")

# Invalid data
ExportError("Failed to serialize data to JSON", export_format="json")

# I/O error
ExportError("Failed to write CSV file", export_format="csv", file_path="/path/file.csv")

# Parse error
ExportError("Invalid JSON file", export_format="json", file_path="/path/file.json")
```

## Tips & Tricks

1. **Pretty Print for Inspection**: Use `pretty=True` when first exporting for debugging
2. **CSV for Analysis**: Export to CSV for use in Excel or Jupyter notebooks
3. **Batch Export**: Create helper function for exporting multiple results
4. **Validation**: Check `r_squared` value (>0.9 is good fit)
5. **Consistency**: Check `statistics.is_consistent()` for reliable measurements

## Performance Notes

- **JSON**: Best for complete preservation, larger file size
- **CSV**: Best for tabular analysis, smaller file size
- **Load**: Both formats load quickly for typical datasets
- **Memory**: In-memory operations suitable for typical profiling data

## Related Classes

- `ProfileResult`: Main data container (from `complexity-profiler.analysis.profiler`)
- `PerformanceMetrics`: Individual run metrics (from `complexity-profiler.analysis.metrics`)
- `Statistics`: Statistical analysis (from `complexity-profiler.analysis.statistics`)
- `ExportError`: Exception handling (from `complexity-profiler.utils.exceptions`)
