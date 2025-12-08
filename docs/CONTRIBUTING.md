# Contributing to Big-O Complexity Analyzer

Thanks for considering contributing. Here's what you need to know.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- pip

Fork and clone:
```bash
git clone https://github.com/yourusername/complexity-profiler.git
cd complexity-profiler
```

Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Development Workflow

### Code Style

The project uses:

- Black for formatting (line length 100)
- isort for import sorting
- mypy for type checking (strict mode)
- flake8 and pylint for linting
- bandit for security scanning

Run all checks before committing:
```bash
# Format code
black complexity-profiler tests

# Sort imports
isort complexity-profiler tests

# Type check
mypy complexity-profiler

# Lint
flake8 complexity-profiler tests
pylint complexity-profiler
```

Or let pre-commit handle it automatically:
```bash
pre-commit run --all-files
```

### Testing

Aim for 85%+ code coverage. New features need tests.

Run tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=complexity-profiler --cov-report=html

# Run specific test file
pytest tests/unit/test_algorithms.py

# Run tests in parallel
pytest -n auto
```

Test structure:
- **Unit tests**: `tests/unit/` - Test individual components
- **Integration tests**: `tests/integration/` - Test component interactions
- **Fixtures**: `tests/conftest.py` - Shared test fixtures

### Type Hints

All code must include full type hints:

```python
def profile_algorithm(
    algorithm: Algorithm[int],
    sizes: list[int],
    runs: int = 10
) -> ProfileResult:
    """Function with complete type hints."""
    ...
```

Run type checking:
```bash
mypy complexity-profiler
```

### Docstrings

Public APIs need docstrings in Google style:

```python
def analyze_complexity(data: list[int], algorithm: str) -> ComplexityClass:
    """
    Analyze the complexity of an algorithm.

    Args:
        data: Input data for analysis
        algorithm: Name of the algorithm to analyze

    Returns:
        Detected complexity class

    Raises:
        AlgorithmError: If algorithm execution fails
        ValueError: If data is empty

    Example:
        >>> result = analyze_complexity([1, 2, 3], "merge_sort")
        >>> print(result)
        ComplexityClass.LINEARITHMIC
    """
    ...
```

## Adding New Features

### Adding a New Algorithm

Create the algorithm class in the right module (sorting.py, searching.py, or graph.py).

Implement the Algorithm protocol:
   ```python
   from complexity-profiler.algorithms.base import Algorithm, AlgorithmMetadata, ComplexityClass
   from typing import TypeVar

   T = TypeVar('T', bound=Comparable)

   class MyNewSort(Generic[T]):
       """My new sorting algorithm."""

       @property
       def metadata(self) -> AlgorithmMetadata:
           return AlgorithmMetadata(
               name="My New Sort",
               category="sorting",
               expected_complexity=ComplexityClass.LINEARITHMIC,
               space_complexity="O(n)",
               stable=True,
               in_place=False,
               description="Description of the algorithm"
           )

       def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
           """Execute the sorting algorithm."""
           # Implementation with metrics collection
           collector.record_comparison()
           collector.record_swap()
           ...
   ```

Add it to the CLI registry in `complexity-profiler/cli/main.py`:
```python
ALGORITHMS = {
    ...
    "my_new_sort": MyNewSort(),
}
```

Write tests in `tests/unit/`:
```python
class TestMyNewSort:
    def test_sorts_correctly(self):
        ...

    def test_handles_edge_cases(self):
        ...

    def test_metrics_collected(self):
        ...
```

Update docs (algorithms.md and README.md).

### Adding a New Exporter

Create a new file in `complexity-profiler/export/` with an export function that takes `ProfileResult` and `Path`. Add it to the CLI in main.py and write tests in test_export.py.

## Pull Request Process

Create a feature branch:
```bash
git checkout -b feature/my-new-feature
```

Make your changes and run all checks:
```bash
# Format and lint
black complexity-profiler tests
isort complexity-profiler tests
flake8 complexity-profiler tests
mypy complexity-profiler

# Test
pytest --cov=complexity-profiler

# Or use pre-commit
pre-commit run --all-files
```

Commit with clear messages:
```bash
git commit -m "Add new sorting algorithm: TimSort

- Implement TimSort with hybrid merge/insertion approach
- Add unit tests
- Update documentation"
```

Push and create a PR:
```bash
git push origin feature/my-new-feature
```

PRs need:
- Passing CI checks
- No decrease in code coverage
- At least one approving review
- Clear description of changes
- Updated docs if applicable

## Code Review Guidelines

When reviewing code, check:

- Algorithm implementation is correct
- Test coverage is good
- Type hints are complete
- Docstrings are clear
- Error handling makes sense
- No obvious performance issues
- Thread safety if applicable

## Reporting Issues

Include Python version, OS, steps to reproduce, expected vs actual behavior, and any error messages. Use the issue templates in `.github/ISSUE_TEMPLATE/`.

## Code of Conduct

Be respectful and inclusive. We're all here to learn and improve.

## Questions?

- Open an issue for feature requests
- Use discussions for questions
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
