# Tests

The suite covers algorithm correctness, operation counts, timing collection,
numerical fitting, exports and CLI behavior. Run it from the repository root
after installing the [development dependencies](../docs/CONTRIBUTING.md).

```bash
pytest
pytest tests/unit/test_curve_fitting.py
pytest tests/integration/test_outputs.py
pytest -m "not slow"
```

The default run enforces 85% coverage and writes `htmlcov/` and `coverage.xml`.
A targeted run may fall below the full-suite coverage threshold; use
`--no-cov` when checking an individual case during development, then run the
full suite before submitting a behavior change.

## Where to add cases

| Area | Files |
| --- | --- |
| Algorithms and operation counts | `unit/test_algorithms.py`, `unit/test_sorting.py`, `test_sorting_quick.py`, `test_phase4.py` |
| Metrics and statistics | `unit/test_metrics.py`, `unit/test_statistics.py`, `test_phase1.py` |
| Fitting and invalid inputs | `unit/test_curve_fitting.py`, `integration/test_fit_evidence.py` |
| CLI and public imports | `unit/test_cli_validation.py`, `unit/test_public_api.py` |
| Profiling workflows | `integration/test_profiler.py` |
| JSON, CSV and charts | `integration/test_outputs.py` |

Fixtures live in `conftest.py`. Use deterministic, independently derived data
for fitting and arithmetic assertions. Reserve actual clock tests for timing
collection; short wall-clock runs should not have to return a particular Big-O
label. Hypothesis tests exercise algorithm properties such as sorting and
preserving the input values.

Chart tests select Matplotlib's noninteractive `Agg` backend before collection.
They inspect figure coordinates and exported PNGs without opening desktop
windows. Normal package users retain their own backend selection.

The [CI workflow](../.github/workflows/ci.yml) runs the suite on Linux, macOS and
Windows with Python 3.10–3.12. It also checks formatting, types, Bandit and a built
wheel outside the source directory. Consult its run results for a specific revision.
