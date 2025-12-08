# Big-O Complexity Analyzer

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A tool for analyzing algorithm complexity empirically. Runs algorithms on different input sizes, fits curves to the runtime data, and tells you the actual Big-O class.

## What it does

Takes an algorithm, runs it against varying input sizes, collects timing and operation metrics, then uses curve fitting to determine the complexity class. Supports sorting, searching, and graph algorithms.

Features:
- Automatic Big-O detection via curve fitting (R-squared metrics)
- Statistical analysis across multiple runs (mean, median, std dev, confidence intervals)
- Tracks comparisons, swaps, array accesses, and recursive depth
- CLI with Rich formatting
- Plots with matplotlib/seaborn
- Export to JSON/CSV
- Type hints throughout, validated with MyPy

## Algorithms

**Sorting:**
- Merge Sort, Quick Sort, Heap Sort
- Bubble Sort, Insertion Sort, Selection Sort

**Searching:**
- Binary Search, Linear Search
- Jump Search, Interpolation Search

**Graph:**
- BFS, DFS, Dijkstra (framework is extensible)

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/yourusername/complexity-profiler.git
cd complexity-profiler
pip install -r requirements.txt
pip install -e .
```

For development:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Usage

Analyze a single algorithm:

```bash
complexity-profiler analyze MergeSort
complexity-profiler analyze QuickSort --sizes 100,500,1000,5000,10000
complexity-profiler analyze BubbleSort --runs 20
complexity-profiler analyze InsertionSort --chart growth_curve.png
```

Compare algorithms:

```bash
complexity-profiler compare MergeSort QuickSort HeapSort
complexity-profiler compare MergeSort QuickSort --sizes 1000,5000,10000 --chart comparison.png
```

Export results:

```bash
complexity-profiler analyze MergeSort --export results.json
complexity-profiler analyze QuickSort --export results.csv --chart chart.png
```

List available algorithms:

```bash
complexity-profiler list
complexity-profiler list --category sorting
```

## Algorithm Reference

### Sorting

| Algorithm | Average | Worst | Space | Stable | In-Place |
|-----------|---------|-------|-------|--------|----------|
| Merge Sort | O(n log n) | O(n log n) | O(n) | Yes | No |
| Quick Sort | O(n log n) | O(n²) | O(log n) | No | Yes |
| Heap Sort | O(n log n) | O(n log n) | O(1) | No | Yes |
| Insertion Sort | O(n²) | O(n²) | O(1) | Yes | Yes |
| Bubble Sort | O(n²) | O(n²) | O(1) | Yes | Yes |
| Selection Sort | O(n²) | O(n²) | O(1) | No | Yes |

### Searching

| Algorithm | Average | Worst | Space | Notes |
|-----------|---------|-------|-------|-------|
| Binary Search | O(log n) | O(log n) | O(1) | Requires sorted array |
| Linear Search | O(n) | O(n) | O(1) | Works on any array |
| Jump Search | O(√n) | O(√n) | O(1) | Requires sorted array |
| Interpolation Search | O(log log n) | O(n) | O(1) | Requires sorted array, uniform distribution |

## Configuration

Optional. Create `config.toml` in your project directory or at `~/.config/complexity-profiler/config.toml`:

```toml
[general]
default_runs = 10
default_sizes = [100, 500, 1000, 5000, 10000]
random_seed = null  # null for random, integer for reproducibility

[profiling]
warmup_runs = 2
timeout_seconds = 300
min_run_time = 0.001

[visualization]
theme = "seaborn-v0_8-darkgrid"
default_format = "png"
dpi = 300
show_confidence_intervals = true
confidence_level = 0.95

[export]
default_format = "json"
pretty_print = true
include_raw_data = false

[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file = "complexity-profiler.log"

[algorithms.sorting]
enabled = ["merge_sort", "quick_sort", "heap_sort", "insertion_sort", "bubble_sort", "selection_sort"]

[algorithms.searching]
enabled = ["binary_search", "linear_search", "jump_search", "interpolation_search"]
```

You can also override settings with environment variables:

```bash
export BIGOCOMPLEXITY_RUNS=20
export BIGOCOMPLEXITY_LOG_LEVEL=DEBUG
export BIGOCOMPLEXITY_SEED=42
```

## Development

Set up a dev environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pre-commit install
```

Run quality checks:

```bash
black complexity-profiler/
isort complexity-profiler/
mypy complexity-profiler/
flake8 complexity-profiler/
pylint complexity-profiler/
bandit -r complexity-profiler/

# Or run everything:
pre-commit run --all-files
```

## Testing

Run tests:

```bash
pytest
pytest --cov=complexity-profiler --cov-report=html
pytest tests/unit/test_sorting.py
pytest -v
pytest -n auto
```

Tests are organized under `tests/unit/` and `tests/integration/`.

## Project Structure

```
complexity-profiler/
├── algorithms/       # Algorithm implementations
├── analysis/         # Profiling, metrics, curve fitting
├── visualization/    # Plotting and charts
├── cli/              # Command-line interface
├── export/           # JSON/CSV exporters
├── config/           # Settings management
└── utils/            # Exceptions, logging

tests/
├── unit/
└── integration/
```

## Examples

Analyzing merge sort:

```bash
complexity-profiler analyze MergeSort --sizes 100,500,1000,5000,10000 --runs 15 --verbose
```

Output will show timing, comparisons, swaps, and the detected complexity (e.g., O(n log n) with R² = 0.9987).

Comparing multiple algorithms:

```bash
complexity-profiler compare MergeSort QuickSort HeapSort --chart comparison.png
```

Full pipeline with export:

```bash
complexity-profiler analyze QuickSort --runs 20 --export results.json --chart growth.png
```

The JSON export includes algorithm metadata, performance metrics, statistical analysis, and the detected complexity class.

## Contributing

PRs welcome. Before submitting:

- Run `pytest` (all tests should pass)
- Run `mypy complexity-profiler/` (no type errors)
- Run `black` and `isort`
- Add tests for new features
- Keep test coverage above 90%

## Architecture

Uses protocol-based design (structural subtyping), strategy pattern for algorithms, dependency injection, and immutable dataclasses. Fully typed with generics and type hints.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Dependencies

Built with NumPy, SciPy, matplotlib, seaborn, pandas, Click, Rich, and pytest.

## Notes

Empirical results vary based on hardware, system load, and input characteristics. Run multiple iterations for statistical significance. This tool is meant for educational and analytical purposes - use it alongside theoretical complexity analysis, not as a replacement.
