# Export quick reference

Given a `ProfileResult` named `result`:

```python
from pathlib import Path
from complexity_profiler import (
    export_metrics_summary_csv,
    export_to_csv,
    export_to_json,
    load_from_csv,
    load_from_json,
)

export_to_json(result, Path("profile.json"))
export_to_json(result, Path("profile-compact.json"), pretty=False)
export_to_csv(result, Path("profile.csv"))
export_metrics_summary_csv(result, Path("summary.csv"))

restored = load_from_json(Path("profile.json"))
table = load_from_csv(Path("profile.csv"), result.algorithm_name, result.category)
```

JSON preserves runs, nested statistics, fit scores and notes. The per-size CSV
contains aggregates; the summary CSV contains one algorithm/fit row. CSV loading
returns a DataFrame, not a reconstructed profile. Functions create missing parent
directories and report failures through `ExportError`.

See the [full guide](../EXPORT_GUIDE.md) for a runnable profiling example and
column names.
