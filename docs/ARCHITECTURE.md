# Architecture

The profiler separates algorithm execution from measurement, fitting and output.
An algorithm receives an input list and a metrics collector; the profiler repeats
that operation for each requested size, summarizes the measurements and fits the
mean timings. The result can then be displayed, plotted or exported.

```text
Input generator → Algorithm + collector → Per-run metrics
                                              ↓
                                   Per-size statistics
                                              ↓
                                      Growth-curve fit
                                              ↓
                                        ProfileResult
                                              ↓
                                  Terminal / charts / files
```

## Modules

All paths below are inside `complexity_profiler/`.

| Module | Responsibility |
| --- | --- |
| `algorithms/base.py` | `Algorithm`, `MetricsCollector`, `AlgorithmMetadata` and growth-class definitions. |
| `algorithms/sorting.py`, `searching.py`, `graph.py` | Instrumented algorithm implementations. |
| `data/generators.py`, `graph_generators.py` | Input distributions and graph structures. |
| `analysis/metrics.py` | Operation counters and timing. |
| `analysis/statistics.py` | Mean, median, standard deviation, percentiles and variation. |
| `analysis/curve_fitting.py` | Candidate growth functions, normalized least squares and ambiguity handling. |
| `analysis/profiler.py` | Repeated runs, fluent configuration and `ProfileResult`. |
| `visualization/` | Matplotlib charts, themes and Rich terminal formatting. |
| `export/` | JSON serialization and per-size or summary CSV tables. |
| `cli/main.py` | Click commands and the sorting/searching algorithm registry. |
| `config/settings.py` | A separate TOML settings loader for library callers. |

## Algorithm and collector interfaces

Algorithms implement the structural `Algorithm` protocol: a `metadata` property
and `execute(data, collector)` method. They do not need to inherit a base class.
The collector records comparisons, swaps, accesses and recursive calls, so the
same algorithm can be used with different measurement implementations.

`AlgorithmProfiler` accepts a generator through `with_data_generator`, sizes
through `with_sizes` and repetitions through `with_runs`. It copies generated
inputs before execution, performs warmups, collects runs and computes statistics.
See [custom experiments](examples/advanced_analysis.md) for a complete example.

## Fitting

The six candidates are constant, logarithmic, linear, linearithmic, quadratic
and cubic. Each model has an intercept and a nonnegative growth coefficient.
Exponential and factorial names exist in `ComplexityClass` but are not fitted.

Inputs must be one-dimensional, finite and equal in length, with at least three
distinct positive sizes and nonnegative timings containing a positive value.
Invalid inputs raise `ValueError`. Exactly flat positive samples use the constant
model. Other samples are normalized before least squares, making the score
independent of timing units. Polynomial bases scale sizes before taking powers;
the linearithmic basis retains the original logarithm. Public curve functions
keep their original coefficient units.

A candidate must meet `min_r_squared` (0.8 by default). A second candidate within
0.02 R² and twice the best residual error makes the result ambiguous. A
64-machine-epsilon residual floor handles numerical equality. An unrepresentable
candidate or an ambiguous/poor fit raises `InconclusiveFitError` in the direct
fitter API. The exception is a `RuntimeError` subclass and carries candidate scores.

The profiler catches that exception and returns a result whose
`empirical_complexity` is `Inconclusive`. It retains the measurements, candidate
scores, best score and reason in `notes`. Timing noise, input distribution and
instrumentation overhead all affect the sampled trend; the ambiguity heuristic
is not a statistical confidence interval.

## Results and output

`ProfileResult` holds algorithm metadata, input sizes, per-run `PerformanceMetrics`,
per-size `Statistics`, the empirical label, R², candidate scores, timestamp and
notes. `to_dict()` recursively serializes the nested dataclasses; `get_summary()`
returns the shorter algorithm/fit summary.

JSON restores a `ProfileResult`, including nested dataclasses and integer size
keys. CSV contains aggregates and loads as a pandas DataFrame. Charts consume
the same result object. See the [export guide](EXPORT_GUIDE.md) for the fields
preserved by each format.

## Configuration and extension

The current CLI takes configuration from its command-line options. It does not
load `config.toml` or apply the `Settings` dataclasses. For library experiments,
configure `AlgorithmProfiler` directly. The separate settings loader recognizes
`BIGOCOMPLEXITY_RUNS` and `BIGOCOMPLEXITY_LOG_LEVEL`; callers must apply loaded
settings themselves. TOML has no `null` value, so omit optional values instead.

To add an algorithm, implement the protocol, instrument its operations and add
correctness tests. Register list-based algorithms in `cli/main.py` if they should
be available from the CLI. Graph algorithms currently use the Python API. Keep
expected theoretical complexity in metadata separate from measured fit results.
