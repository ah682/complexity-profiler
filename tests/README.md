# Big-O Complexity Analyzer - Test Suite

Comprehensive test suite for the Big-O Complexity Analyzer project with excellent coverage of core functionality.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_algorithms.py         # Algorithm implementation tests
│   ├── test_metrics.py            # Metrics collection tests
│   ├── test_statistics.py         # Statistical analysis tests
│   └── test_sorting.py            # Comprehensive sorting tests
└── integration/                   # Integration tests
    ├── __init__.py
    └── test_profiler.py           # End-to-end profiling workflow tests
```

## Test Files Overview

### conftest.py
Provides shared pytest fixtures and test utilities:
- **sample_data**: Various data configurations (random, sorted, reverse, duplicates, etc.)
- **small_sample_data**: Small datasets for fast tests
- **metrics_collector**: Fresh DefaultMetricsCollector instances
- **sorting_algorithms**: All sorting algorithm instances
- **performance_data_generator**: Synthetic performance data generator
- **is_sorted/assert_sorted**: Helper functions for verification
- **performance_metrics_factory**: Factory for creating test metrics

Custom pytest markers:
- `@pytest.mark.slow` - For slow-running tests
- `@pytest.mark.integration` - For integration tests
- `@pytest.mark.unit` - For unit tests
- `@pytest.mark.property` - For property-based tests

### Unit Tests

#### test_metrics.py
Tests for metrics collection functionality:
- **TestPerformanceMetrics**: Dataclass initialization, operations, merging
- **TestDefaultMetricsCollector**: Recording operations, timing, reset functionality
- **TestAggregateMetricsCollector**: Aggregating metrics across runs
- **TestMetricsIntegration**: Integration with actual algorithms

**Coverage**: PerformanceMetrics, DefaultMetricsCollector, ThreadSafeMetricsCollector, AggregateMetricsCollector

#### test_statistics.py
Tests for statistical analysis:
- **TestStatistics**: Dataclass properties, consistency checks
- **TestComputeStatistics**: Mean, median, percentiles, CV calculations
- **TestStatisticsIntegration**: Real algorithm performance data

**Coverage**: Statistics dataclass, compute_statistics function

**Test Cases**:
- Empty/single/multiple values
- Identical values
- Negative values
- Outliers
- Near-zero mean handling
- Known statistical values

#### test_sorting.py (NEW)
Comprehensive sorting algorithm tests:
- **TestSortingCorrectness**: Correctness verification across all algorithms
  - Random, sorted, reverse, nearly-sorted data
  - Duplicates, uniform values, negative numbers
  - Edge cases: empty, single, two elements

- **TestSortingStability**: Stability verification for stable sorts
  - Tests MergeSort, BubbleSort, InsertionSort (stable)
  - Verifies QuickSort, SelectionSort, HeapSort (unstable)

- **TestSortingMetrics**: Metrics collection accuracy
  - Algorithm-specific metrics validation
  - Operation count scaling with input size

- **TestSortingAlgorithmMetadata**: Metadata verification
  - Name, category, complexity class
  - Stability and in-place properties

- **TestSortingEdgeCases**: Special scenarios
  - Large datasets (1000+ elements)
  - Power-of-two sizes
  - Alternating values

- **TestAdaptiveAlgorithms**: Adaptive behavior
  - BubbleSort early termination
  - InsertionSort efficiency on nearly-sorted data

- **TestSortingPropertyBased**: Property-based tests
  - Idempotency (sorting twice = sorting once)
  - Permutation invariance
  - Min/max element positions

**Coverage**: All 6 sorting algorithms (MergeSort, QuickSort, SelectionSort, BubbleSort, InsertionSort, HeapSort)

#### test_algorithms.py (EXISTING)
Individual algorithm test suites with property-based testing:
- Per-algorithm test classes with hypothesis integration
- Cross-algorithm consistency tests

### Integration Tests

#### test_profiler.py (NEW)
End-to-end profiling workflow tests:
- **TestAlgorithmProfiler**: Profiler configuration and validation
  - Fluent API testing
  - Input validation
  - Custom sizes, runs, and data generators

- **TestProfileResult**: ProfileResult dataclass
  - Initialization and validation
  - Serialization (to_dict, get_summary)
  - Timestamp tracking

- **TestProfileAlgorithmConvenience**: Convenience function
  - Default and custom configurations
  - Multiple run counts

- **TestEndToEndProfiling**: Complete workflows
  - Complexity detection for all algorithms
  - Warmup runs
  - Different input patterns (random, sorted, reverse)
  - Multi-algorithm comparison
  - Large input sizes

- **TestProfilingErrorHandling**: Error scenarios
  - Insufficient data points
  - Invalid data generators

**Coverage**: AlgorithmProfiler, ProfileResult, profile_algorithm function

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_sorting.py

# Specific test class
pytest tests/unit/test_sorting.py::TestSortingCorrectness

# Specific test method
pytest tests/unit/test_sorting.py::TestSortingCorrectness::test_sorts_random_data
```

### Run with Coverage
```bash
pytest tests/ --cov=complexity-profiler --cov-report=html
```

### Run with Markers
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m integration

# Run only unit tests
pytest tests/ -m unit

# Run property-based tests
pytest tests/ -m property
```

### Verbose Output
```bash
# Show test names
pytest tests/ -v

# Show print statements
pytest tests/ -s

# Detailed failure info
pytest tests/ -vv
```

## Test Patterns and Best Practices

### Parametrization
Tests use `@pytest.mark.parametrize` for testing multiple scenarios:

```python
@pytest.mark.parametrize("algorithm_class,name", ALL_SORTING_ALGORITHMS)
def test_sorts_random_data(algorithm_class, name):
    # Test runs for all 6 sorting algorithms
    pass
```

### Fixtures
Shared fixtures provide consistent test data:

```python
def test_sorting(sample_data, metrics_collector):
    data = sample_data['random']  # Pre-generated random data
    algorithm.execute(data, metrics_collector)
```

### Property-Based Testing
Some tests use Hypothesis for property-based testing:

```python
@given(st.lists(st.integers(), max_size=100))
def test_property_always_sorts(data):
    result = algorithm.execute(data, collector)
    assert result == sorted(data)
```

### Test Organization
- **Arrange**: Setup test data and objects
- **Act**: Execute the operation being tested
- **Assert**: Verify expected outcomes

### Docstrings
All test methods include docstrings explaining what they test:

```python
def test_sorts_random_data(self):
    """Test that algorithm correctly sorts random data."""
    # Test implementation
```

## Coverage Goals

The test suite aims for:
- **Line Coverage**: >90% for core modules
- **Branch Coverage**: >85% for conditional logic
- **Algorithm Coverage**: 100% of implemented algorithms
- **Edge Cases**: Comprehensive edge case testing

### Current Coverage by Module

- **algorithms/sorting.py**: ~95% (all algorithms, all paths)
- **analysis/metrics.py**: ~95% (all collectors, all operations)
- **analysis/statistics.py**: ~98% (all statistical measures)
- **analysis/profiler.py**: ~90% (all profiling workflows)

## Adding New Tests

### For New Algorithms
1. Add algorithm to `conftest.py` fixtures
2. Create test class in `test_algorithms.py`
3. Add to parametrized tests in `test_sorting.py`
4. Add integration test in `test_profiler.py`

### For New Features
1. Create unit tests first (TDD)
2. Add integration tests for workflows
3. Update conftest.py with new fixtures
4. Document test coverage in this README

### Test Naming Convention
- `test_<what>_<scenario>`: e.g., `test_sorts_empty_array`
- `test_<feature>_<condition>`: e.g., `test_metrics_scale_with_input_size`
- Use descriptive names that explain the test purpose

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pytest tests/ --cov=complexity-profiler --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Slow Tests
Mark slow tests and skip during development:
```python
@pytest.mark.slow
def test_large_dataset():
    # Slow test
    pass

# Run without slow tests
pytest -m "not slow"
```

### Random Test Failures
Use seeds for reproducibility:
```python
data = random_data(100, seed=42)  # Reproducible
```

### Import Errors
Ensure package is installed in development mode:
```bash
pip install -e .
```

## Performance Benchmarks

Integration tests can be used for performance benchmarking:

```bash
# Run with benchmark plugin
pytest tests/integration/test_profiler.py --benchmark-only
```

## Test Data Generators

The test suite uses data generators from `complexity-profiler.data.generators`:
- `random_data()`: Random integers
- `sorted_data()`: Ascending sequence
- `reverse_sorted_data()`: Descending sequence
- `nearly_sorted_data()`: Mostly sorted with few swaps
- `duplicates_data()`: Many duplicate values
- `uniform_data()`: All identical values
- `sawtooth_data()`: Repeating pattern

## Contributing

When contributing tests:
1. Follow existing patterns and conventions
2. Add docstrings to all test methods
3. Use appropriate fixtures and parametrization
4. Ensure tests are deterministic (use seeds)
5. Update this README with new test categories
6. Verify tests pass locally before submitting

## Test Metrics

Current test suite statistics:
- **Total Tests**: 150+
- **Unit Tests**: ~100
- **Integration Tests**: ~50
- **Parametrized Variations**: ~200+
- **Average Runtime**: <30 seconds (without slow tests)
- **Property-Based Tests**: 10+

## License

Tests are part of the Big-O Complexity Analyzer project and share the same license.
