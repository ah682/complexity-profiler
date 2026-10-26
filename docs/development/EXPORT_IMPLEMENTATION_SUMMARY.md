# Export implementation notes

Exports consume the same `ProfileResult` used by charts and terminal output.
`ProfileResult.to_dict()` converts nested metric/statistic dataclasses, integer
size keys and timestamps into JSON-compatible values.

`json_exporter.py` handles NumPy scalars and arrays through `NumpyEncoder` and
restores nested `PerformanceMetrics` and `Statistics` on load. `csv_exporter.py`
creates per-size aggregate tables or a separate one-row summary. CSV loading
returns a DataFrame and checks its algorithm/category against the caller's values.
Both modules wrap file and conversion failures in `ExportError`.

The functions are re-exported from `complexity_profiler.export` and the package
root. The [export guide](../EXPORT_GUIDE.md) is the maintained API reference;
[output integration tests](../../tests/integration/test_outputs.py) cover its
round trips, columns and error cases.
