# Complexity Profiler

A Python toolkit for exploring how algorithms behave as their inputs grow. Run
sorting and searching algorithms, collect timings and operation counts, compare
growth curves, and export the results as charts, JSON or CSV. Graph algorithms
are also available through the Python API.

Use it to compare input patterns, study algorithms and inspect measured trends.
Curve fits describe the sampled range; they do not prove an algorithm's Big-O
class. When the measurements cannot distinguish the candidates, the profiler
returns `Inconclusive` and keeps the raw data and scores for inspection.

## Install and run

Python 3.10 or newer is required.

```bash
git clone https://github.com/ah682/complexity-profiler.git
cd complexity-profiler
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
complexity-profiler list-algorithms
complexity-profiler analyze merge_sort --sizes 100,500,1000,5000 --runs 10 --no-visualize
complexity-profiler analyze insertion_sort --sizes 50,100,200,400 --runs 5 --no-visualize --export json --output result.json
complexity-profiler compare merge_sort heap_sort --sizes 100,500,1000 --runs 5 --save-chart comparison.png
```

The installed command works outside the source directory. Use `--help` on any
command for options. `--save-chart chart.png` saves a chart for `analyze`; use
`--no-visualize` for a terminal-only run. JSON preserves raw measurements and fit
scores; the CSV export contains per-size aggregates, rather than a full round trip.

## Use as a library

```python
from complexity_profiler.analysis.curve_fitting import fit_to_complexity, InconclusiveFitError

sizes = [1, 2, 4, 8, 16, 32]
times = [1e-6, 2e-6, 4e-6, 8e-6, 16e-6, 32e-6]
try:
    complexity, score = fit_to_complexity(sizes, times, min_r_squared=0.8)
    print(complexity, score)  # O(n), 1.0 for these controlled samples
except InconclusiveFitError as error:
    print(error, error.candidate_scores)
```

`ComplexityFitter.candidate_scores` exposes all six candidate scores after a fit.
`AlgorithmProfiler` keeps raw metrics, per-size statistics and the best candidate
score even when its `ProfileResult.empirical_complexity` is `Inconclusive`; the
reason appears in `notes` and scores in `candidate_scores`. Successful direct
fitter calls retain the `(ComplexityClass, r_squared)` return type. Inconclusive
direct fits raise `InconclusiveFitError`, a `RuntimeError` subclass.

## Interpreting a fit

- Supply at least three **distinct positive** sizes and matching one-dimensional,
  finite, nonnegative timings; at least one timing must be positive. Invalid data
  raises `ValueError`. Thresholds must be finite and between 0 and 1.
- Models are constant, logarithmic, linear, linearithmic, quadratic and cubic,
  each with an intercept and a nonnegative growth coefficient. Exponential and
  factorial are enum labels, not supported fit candidates.
- The fitter normalizes timings and growth bases to avoid overflow and underflow.
  Changing timing units does not change the score. Exactly flat positive samples
  receive a constant fit; small variations are still fitted.
- A candidate must meet `min_r_squared`. Candidates within 0.02 R² **and** twice
  the best residual error are considered ambiguous, with a 64-machine-epsilon
  residual floor for numerical equality. This ambiguity rule is a heuristic,
  not a confidence interval or significance test.
- R² describes the sampled range only. Expand the size range, inspect candidate
  scores, and consider deterministic operation counts before interpreting growth.
  Instrumentation adds overhead. Repeated runs reduce some noise but do not remove
  environmental bias. Charts show observations and their mean trend.
- Search algorithms use the first list item as the target and remaining items as
  the search array. Ordered search algorithms require sorted data; the CLI's generic
  data generator does not establish an average-case search experiment.

## Reproducible chart

```bash
python examples/controlled_growth.py
```

This writes `controlled-growth.png` and `controlled-growth.json` from synthetic
linear measurements, giving a repeatable example of fitting and plotting.

![Controlled synthetic linear measurements](docs/assets/controlled-growth.png)

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest
black --check --workers 1 complexity_profiler tests
isort --check-only complexity_profiler tests
flake8 complexity_profiler tests
mypy complexity_profiler
bandit -r complexity_profiler
python -m build
```

Alternatively, `uv sync --extra dev` installs the locked development environment;
run checks with `uv run`. CI checks formatting, types, security and an 85% coverage
threshold across Linux, macOS and Windows with Python 3.10–3.12. Bandit's B311
annotations cover pseudorandom input generation and algorithm pivots.

Bug reports should include Python/package versions, OS, input sizes, timing units,
raw data and candidate scores. For fitting changes, add deterministic regressions
with independently derived expected results across timing scales. Keep real timing
smoke tests separate from model correctness tests. See the
[contribution guide](docs/CONTRIBUTING.md) and [test guide](tests/README.md).

MIT licensed; see [LICENSE](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for release
notes and the [documentation index](docs/README.md) for examples, architecture and
export details.
