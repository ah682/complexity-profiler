# Contributing

Bug reports and focused improvements are welcome. Include Python and package
versions, the OS, input sizes, data distribution, expected behavior and the actual
output. For fitting problems, attach raw timings, their units and candidate scores.

## Set up development

Use Python 3.10 or newer:

```bash
git clone https://github.com/ah682/complexity-profiler.git
cd complexity-profiler
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
Alternatively, use `uv sync --extra dev` and prefix commands with `uv run`.

## Checks

```bash
black --check --workers 1 complexity_profiler tests
isort --check-only complexity_profiler tests
flake8 complexity_profiler tests
mypy complexity_profiler
bandit -r complexity_profiler
pytest
python -m build
```

Black and isort use a 100-character line length. Pytest enforces 85% coverage and
writes terminal, HTML and XML reports. The [CI workflow](../.github/workflows/ci.yml)
also installs the wheel and exercises the CLI outside the checkout. Optional
pre-commit hooks can be installed with `pre-commit install`.

## Changes to algorithms and fitting

Implement the `Algorithm` protocol in the appropriate module under
`complexity_profiler/algorithms/`. Record operations through the supplied
collector, add correctness cases and update the [algorithm reference](algorithms.md).
The CLI registry is in `complexity_profiler/cli/main.py`.

For fitting changes, use hand-derived data with known expected scores. Include
invalid inputs, ambiguous candidates, flat samples and different timing units.
Keep these tests deterministic. Real-clock tests should check measurement
collection, without requiring a particular fitted class from a short timing run.

An exporter should accept a `ProfileResult` and `Path`, preserve its documented
fields and raise `ExportError` with enough context to diagnose failures. Add
round-trip and failure-path cases to `tests/integration/test_outputs.py`.

Describe the problem, final behavior and checks in the pull request. Update examples
when a public interface changes. Contributions use the project's [MIT licence](../LICENSE).
