# Exporting results

Use JSON to save a profile for later inspection. Use CSV to compare per-size
aggregates in a spreadsheet or pandas. Both formats accept a `ProfileResult`
and create missing parent directories.

## Profile and export

```python
from pathlib import Path

from complexity_profiler import (
    AlgorithmProfiler,
    export_metrics_summary_csv,
    export_to_csv,
    export_to_json,
    load_from_csv,
    load_from_json,
)
from complexity_profiler.algorithms.sorting import MergeSort
from complexity_profiler.data.generators import random_data

result = (
    AlgorithmProfiler()
    .with_sizes([100, 500, 1000, 5000])
    .with_runs(5)
    .with_data_generator(random_data)
    .profile(MergeSort())
)

export_to_json(result, Path("results/merge-sort.json"))
export_to_csv(result, Path("results/merge-sort.csv"))
export_metrics_summary_csv(result, Path("results/merge-sort-summary.csv"))

restored = load_from_json(Path("results/merge-sort.json"))
table = load_from_csv(Path("results/merge-sort.csv"), "Merge Sort", "sorting")
print(restored.candidate_scores)
print(table[["input_size", "stat_mean", "stat_std_dev"]])
```

`examples/export_example.py` demonstrates the formats with constructed sample
measurements. Its supplied fit label is example metadata, not a measured result.

## API

| Function | Result |
| --- | --- |
| `export_to_json(result, file_path, pretty=True)` | JSON document; set `pretty=False` for compact output. |
| `load_from_json(file_path)` | Restored `ProfileResult`. |
| `export_to_csv(result, file_path)` | One row per input size with metrics and statistics. |
| `export_metrics_summary_csv(result, file_path)` | One row containing the algorithm and fit summary. |
| `load_from_csv(file_path, algorithm_name, category)` | DataFrame; checks the first row's algorithm and category against the supplied values. |

These functions are available from both `complexity_profiler` and
`complexity_profiler.export`. `NumpyEncoder` is available from the export package
for callers that need its NumPy scalar/array conversion separately.

## What each format preserves

JSON contains the algorithm name, category, input sizes, all per-run metrics,
per-size statistics, empirical label, R², candidate scores, timestamp and notes.
Size keys become strings and the timestamp becomes ISO text in the file; loading
restores integer keys, the timestamp and nested dataclasses. Inconclusive profiles
keep their explanation and scores.

The per-size CSV includes:

- Identity: `algorithm`, `category`, `input_size`, `num_runs`.
- Mean metrics: `mean_comparisons`, `mean_swaps`, `mean_accesses`,
  `mean_recursive_calls`, `mean_memory_operations`, `mean_execution_time`,
  `mean_total_operations`.
- Timing range: `min_execution_time`, `max_execution_time`.
- Statistics: `stat_mean`, `stat_median`, `stat_std_dev`, `stat_min`, `stat_max`,
  `stat_p25`, `stat_p75`, `stat_cv`.

Sizes without both metrics and statistics are skipped; a profile with no
exportable rows raises `ExportError`. CSV does not preserve individual runs or
candidate scores and cannot reconstruct a `ProfileResult`. The summary CSV uses
`get_summary()` and is separate from the per-size format expected by `load_from_csv`.

## Error handling

```python
from pathlib import Path
from complexity_profiler import load_from_json
from complexity_profiler.utils.exceptions import ExportError

try:
    restored = load_from_json(Path("results/merge-sort.json"))
except ExportError as error:
    print(error)
```

Export errors identify the format, path and underlying failure. Check the path,
permissions and input format before retrying. JSON and CSV are assembled in memory,
so their memory use grows with the profile.

The integration tests in [test_outputs.py](../tests/integration/test_outputs.py)
cover serialization, nested values, CSV columns and generated charts. Run the
full suite with `pytest` as described in the [test guide](../tests/README.md).
