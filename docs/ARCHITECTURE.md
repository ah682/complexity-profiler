# Big-O Complexity Analyzer - Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Patterns](#design-patterns)
3. [Module Structure](#module-structure)
4. [Protocol-Based Design](#protocol-based-design)
5. [Data Flow](#data-flow)
6. [Extension Points](#extension-points)
7. [Component Interactions](#component-interactions)
8. [Type Safety](#type-safety)

---

## System Overview

The Big-O Complexity Analyzer is a professional-grade framework for empirically analyzing algorithm complexity through performance profiling, statistical analysis, and curve fitting. It provides automated Big-O complexity detection, detailed performance metrics, and beautiful visualizations for sorting, searching, and graph algorithms.

### Key Capabilities

- **Empirical Complexity Detection**: Automatically fits measured performance data to theoretical Big-O complexity curves
- **Comprehensive Metrics Tracking**: Captures comparisons, swaps, array accesses, recursive calls, and execution time
- **Statistical Analysis**: Computes mean, median, standard deviation, percentiles, and confidence intervals
- **Professional Visualizations**: Generates growth curves, comparative charts, and distribution plots
- **Extensible Architecture**: Protocol-based design enables easy addition of new algorithms
- **Type-Safe Implementation**: Full type hints with MyPy validation
- **Export Capabilities**: JSON and CSV output with rich metadata

### Core Philosophy

The analyzer separates concerns into distinct layers:

1. **Algorithm Layer**: Individual algorithm implementations following a protocol
2. **Profiling Layer**: Execution, timing, and metrics collection
3. **Analysis Layer**: Statistical analysis and complexity curve fitting
4. **Visualization Layer**: Chart generation and display
5. **CLI Layer**: User-facing command interface

---

## Design Patterns

### 1. Strategy Pattern

The analyzer uses the Strategy pattern to make algorithms interchangeable and independently variability.

**Implementation**: The `Algorithm` protocol defines a common interface that all algorithms must implement:

```python
class Algorithm(Protocol[T]):
    @property
    def metadata(self) -> AlgorithmMetadata: ...

    def execute(self, data: list[T], collector: 'MetricsCollector') -> list[T]: ...
```

**Benefits**:
- New algorithms can be added without modifying existing code
- Algorithms can be selected at runtime based on requirements
- Easy to swap implementations for testing or comparison

**Example Usage**:
```python
# Different strategies, same interface
merge_sort = MergeSort()
quick_sort = QuickSort()

# Used interchangeably
for algo in [merge_sort, quick_sort]:
    result = profiler.profile(algo)
```

### 2. Builder Pattern

The `AlgorithmProfiler` class implements the Builder pattern for flexible configuration:

```python
profiler = AlgorithmProfiler()
result = (profiler
    .with_sizes([100, 500, 1000])
    .with_runs(10)
    .with_data_generator(my_generator)
    .profile(algorithm))
```

**Benefits**:
- Fluent API for intuitive configuration
- Default sensible values that can be customized
- Clear separation between configuration and execution

**Methods**:
- `with_sizes(sizes)`: Set input sizes to test
- `with_runs(runs)`: Set number of runs per size
- `with_data_generator(generator)`: Set data generation strategy

### 3. Observer Pattern

The metrics collection system uses the Observer pattern to track algorithm operations without intrusive instrumentation.

**Implementation**: Algorithms inject a `MetricsCollector` and call observation methods:

```python
def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
    for i in range(len(data)):
        collector.record_access()  # Observe data access
        for j in range(i):
            collector.record_comparison()  # Observe comparison
            if data[j] > data[i]:
                collector.record_swap()  # Observe swap
```

**Benefits**:
- Metrics collection is decoupled from algorithm logic
- Different collectors can be used (single-threaded, thread-safe, aggregate)
- Minimal performance overhead

**Collector Implementations**:
- `DefaultMetricsCollector`: Basic single-threaded collection
- `ThreadSafeMetricsCollector`: Thread-safe version with locks
- `AggregateMetricsCollector`: Aggregates metrics across multiple runs

### 4. Dependency Injection

The framework extensively uses dependency injection to decouple components:

```python
# Data generators are injected
profiler.with_data_generator(lambda n: list(range(n)))

# Metrics collectors are injected
algorithm.execute(data, collector)

# Settings are injected into components
visualization.apply_theme(settings.visualization.theme)
```

**Benefits**:
- Easy to test components in isolation
- Swap implementations without changing code
- Clear component dependencies

---

## Module Structure

### High-Level Architecture

```
complexity-profiler/
├── algorithms/           # Algorithm implementations and protocols
│   ├── base.py          # Core protocols: Algorithm, MetricsCollector
│   ├── sorting.py       # Sorting algorithms (MergeSort, QuickSort, etc.)
│   ├── searching.py     # Searching algorithms (BinarySearch, LinearSearch, etc.)
│   └── graph.py         # Graph algorithms (extensible)
│
├── analysis/            # Profiling, metrics, and analysis
│   ├── profiler.py      # AlgorithmProfiler - main profiling engine
│   ├── metrics.py       # Metrics collection classes
│   ├── statistics.py    # Statistical analysis utilities
│   └── curve_fitting.py # Complexity curve fitting
│
├── visualization/       # Chart generation and presentation
│   ├── charts.py        # ChartGenerator for plots
│   ├── themes.py        # Visual themes and styling
│   └── formatters.py    # Output formatting utilities
│
├── cli/                 # Command-line interface
│   ├── main.py          # CLI entry point (Click-based)
│   ├── commands/        # Individual command implementations
│   │   └── analyze.py   # Analyze command
│   └── validators.py    # Input validation
│
├── export/              # Data export functionality
│   ├── json_exporter.py # JSON export implementation
│   └── csv_exporter.py  # CSV export implementation
│
├── data/                # Data generation utilities
│   ├── generators.py    # Sorting/searching data generators
│   └── graph_generators.py  # Graph data generators
│
├── config/              # Configuration management
│   ├── settings.py      # Settings loading and management
│   └── defaults.py      # Default configuration values
│
└── utils/               # Utility modules
    ├── exceptions.py    # Custom exception classes
    └── logging_config.py # Logging configuration
```

### Module Responsibilities

#### algorithms/base.py
- **Role**: Define core abstractions
- **Exports**:
  - `Algorithm[T]`: Protocol for all algorithms
  - `MetricsCollector`: Protocol for metrics observation
  - `Comparable`: Protocol for comparable types
  - `ComplexityClass`: Enum of Big-O classes
  - `AlgorithmMetadata`: Dataclass describing algorithm properties

#### algorithms/sorting.py
- **Role**: Implement sorting algorithms
- **Classes**:
  - `MergeSort`: O(n log n) stable divide-and-conquer
  - `QuickSort`: O(n log n) average-case fast partitioning
  - `HeapSort`: O(n log n) in-place heap-based
  - `BubbleSort`: O(n²) simple but stable
  - `InsertionSort`: O(n²) efficient for small/nearly-sorted data
  - `SelectionSort`: O(n²) simple selection-based

#### analysis/profiler.py
- **Role**: Execute profiling workflow
- **Key Classes**:
  - `AlgorithmProfiler`: Main profiling engine with fluent API
  - `ProfileResult`: Complete profiling results dataclass
- **Responsibilities**:
  - Warmup runs for cache priming
  - Multiple runs per size for statistical validity
  - Metrics collection and aggregation
  - Coordinates with curve fitting

#### analysis/metrics.py
- **Role**: Collect and track metrics
- **Key Classes**:
  - `PerformanceMetrics`: Dataclass holding collected metrics
  - `DefaultMetricsCollector`: Basic metrics collection
  - `ThreadSafeMetricsCollector`: Thread-safe variant
  - `AggregateMetricsCollector`: Aggregates multiple runs
- **Tracked Metrics**:
  - Comparisons
  - Swaps
  - Array accesses
  - Recursive calls
  - Execution time
  - Memory operations

#### analysis/statistics.py
- **Role**: Statistical analysis of measurements
- **Key Classes**:
  - `Statistics`: Dataclass for statistical summary
  - `compute_statistics()`: Function to calculate statistics
- **Computed Metrics**:
  - Mean, median
  - Standard deviation
  - Percentiles (25th, 75th)
  - Coefficient of variation

#### analysis/curve_fitting.py
- **Role**: Determine empirical complexity
- **Key Classes**:
  - `ComplexityFitter`: Fits data to complexity curves
- **Supported Complexities**:
  - O(1), O(log n), O(n), O(n log n), O(n²), O(n³)
- **Method**:
  - Non-linear least squares fitting
  - R² goodness-of-fit evaluation
  - Preference for simpler complexities when fit is similar

#### visualization/charts.py
- **Role**: Generate visualizations
- **Key Class**:
  - `ChartGenerator`: Creates matplotlib-based charts
- **Chart Types**:
  - Single algorithm growth curves
  - Algorithm comparison plots
  - Confidence interval bands

#### cli/main.py
- **Role**: CLI entry point
- **Commands**:
  - `analyze`: Analyze single algorithm
  - `compare`: Compare multiple algorithms
  - `list`: List available algorithms
- **Features**:
  - Click-based command structure
  - Rich formatted output
  - Logging configuration
  - Export integration

#### config/settings.py
- **Role**: Configuration management
- **Features**:
  - TOML file loading
  - Environment variable overrides
  - Section-based settings (general, profiling, visualization, export, logging)

---

## Protocol-Based Design

The analyzer leverages Python's protocol-based structural subtyping (PEP 544) for maximum flexibility.

### Core Protocols

#### Algorithm[T] Protocol
```python
@runtime_checkable
class Algorithm(Protocol[T]):
    """Any class implementing these methods is an Algorithm."""

    @property
    def metadata(self) -> AlgorithmMetadata: ...

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]: ...
```

**Key Features**:
- Generic type parameter `T` for type safety
- No explicit inheritance required
- Uses `@runtime_checkable` for isinstance() checks
- Structural typing: "If it quacks like an algorithm..."

#### MetricsCollector Protocol
```python
@runtime_checkable
class MetricsCollector(Protocol):
    """Any class implementing these methods can collect metrics."""

    def record_comparison(self) -> None: ...
    def record_swap(self) -> None: ...
    def record_access(self) -> None: ...
    def record_recursive_call(self) -> None: ...
    def get_metrics(self) -> Any: ...
    def reset(self) -> None: ...
```

**Implementations**:
- `DefaultMetricsCollector`: Single-threaded
- `ThreadSafeMetricsCollector`: Multi-threaded
- `AggregateMetricsCollector`: Multi-run aggregation

#### Comparable Protocol
```python
@runtime_checkable
class Comparable(Protocol):
    """Types that can be compared."""

    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
```

### Benefits of Protocol-Based Design

1. **Structural Typing**: No explicit inheritance needed
2. **Composition Over Inheritance**: Mix and match implementations
3. **Type Safety**: Full type checking with MyPy
4. **Extensibility**: Easy to add new implementations
5. **Testing**: Simple mock implementations for tests
6. **Runtime Flexibility**: Check protocol compliance at runtime

---

## Data Flow

### Complete Request Flow Diagram

```
User
  |
  v (CLI Command)
CLI Layer (main.py)
  |
  +-- Parse Arguments
  +-- Load Configuration
  +-- Create Algorithm Instance
  |
  v
Profiler Layer (profiler.py)
  |
  +-- Setup: with_sizes(), with_runs(), with_data_generator()
  |
  v
Execution Loop (for each input size)
  |
  +-- Generate Test Data
  +-- Warmup Runs (cache priming)
  |
  v
Profiling Runs Loop (N runs per size)
  |
  +-- Create Metrics Collector
  +-- Execute Algorithm (algorithm.execute(data, collector))
  |
  v
Algorithm Execution (sorting.py, searching.py, etc.)
  |
  +-- Perform Algorithm Logic
  +-- Call collector.record_comparison()
  +-- Call collector.record_swap()
  +-- Call collector.record_access()
  +-- Call collector.record_recursive_call()
  |
  v
Metrics Collection (metrics.py)
  |
  +-- Track operation counts
  +-- Record execution time
  +-- Accumulate in PerformanceMetrics
  |
  v
Statistics Computation (statistics.py)
  |
  +-- Compute mean, median, std_dev
  +-- Calculate percentiles
  +-- Compute coefficient of variation
  |
  v
Curve Fitting (curve_fitting.py)
  |
  +-- Attempt fit for each complexity class
  +-- Calculate R² goodness-of-fit
  +-- Select best complexity class
  |
  v
Results (profiler.py -> ProfileResult)
  |
  +-- algorithm_name, category
  +-- input_sizes, metrics_per_size
  +-- statistics_per_size, empirical_complexity
  +-- r_squared, timestamp
  |
  v
Output Processing
  |
  +-- Format Results (formatters.py)
  +-- Generate Visualization (charts.py)
  +-- Export Results (json_exporter.py, csv_exporter.py)
  |
  v
User
```

### Step-by-Step Data Flow Example

```python
# 1. User initiates via CLI
$ complexity-profiler analyze merge_sort --sizes 100,500,1000 --runs 5

# 2. CLI parses arguments and creates profiler
profiler = AlgorithmProfiler()

# 3. Profiler configured with builder pattern
profiler \
    .with_sizes([100, 500, 1000]) \
    .with_runs(5) \
    .with_data_generator(random_generator)

# 4. For each size, multiple runs are executed
for size in [100, 500, 1000]:
    for run in range(5):
        # 5. Create fresh collector and copy data
        collector = DefaultMetricsCollector()
        data_copy = copy.deepcopy(test_data)

        # 6. Execute algorithm with metrics collection
        with collector.timing():
            merge_sort.execute(data_copy, collector)

        # 7. Collect metrics from this run
        metrics = collector.get_metrics()
        # PerformanceMetrics(
        #     comparisons=847,
        #     swaps=523,
        #     accesses=4982,
        #     execution_time=0.0023,
        #     ...
        # )

# 8. Compute statistics across runs per size
for size in input_sizes:
    times = [m.execution_time for m in metrics_per_size[size]]
    stats = compute_statistics(times)
    # Statistics(
    #     mean=0.0024,
    #     std_dev=0.0001,
    #     ...
    # )

# 9. Fit complexity curve
sizes_array = np.array([100, 500, 1000])
mean_times = np.array([0.0024, 0.0122, 0.0289])

fitter = ComplexityFitter()
complexity, r_squared = fitter.fit_complexity(sizes_array, mean_times)
# Returns: (ComplexityClass.LINEARITHMIC, 0.9987)

# 10. Create ProfileResult
result = ProfileResult(
    algorithm_name="Merge Sort",
    category="sorting",
    input_sizes=[100, 500, 1000],
    metrics_per_size={...},
    statistics_per_size={...},
    empirical_complexity="O(n log n)",
    r_squared=0.9987,
    timestamp=datetime.now()
)

# 11. Format and display results
formatted = format_profile_result(result)
console.print(formatted)

# 12. Generate visualization
chart_gen = ChartGenerator()
chart_gen.plot_complexity_curve(result, save_path="merge_sort.png")

# 13. Export results
export_to_json(result, Path("results.json"))
```

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI Layer                              │
│  Parses arguments, loads config, orchestrates workflow          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      Profiler Layer                             │
│  AlgorithmProfiler: Configuration and orchestration             │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼──────────┐     ┌────────────▼──────────┐
│  Algorithm Layer   │     │  Data Generation      │
│  - MergeSort       │     │  - Random integers    │
│  - QuickSort       │     │  - Sorted sequences   │
│  - BubbleSort      │     │  - Reverse sequences  │
│  - etc.            │     │  - Nearly sorted      │
└────────┬───────────┘     └────────────────────────┘
         │
         │ (during execution)
         │
┌────────▼────────────────────────────────────────────────────────┐
│                     Metrics Layer                               │
│  Observers: DefaultMetricsCollector                             │
│  - record_comparison()                                          │
│  - record_swap()                                                │
│  - record_access()                                              │
│  - timing context manager                                       │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ (aggregate per size)
         │
┌────────▼────────────────────────────────────────────────────────┐
│                   Statistics Layer                              │
│  compute_statistics(): Calculate mean, std_dev, percentiles     │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ (fit curve)
         │
┌────────▼────────────────────────────────────────────────────────┐
│                   Curve Fitting Layer                           │
│  ComplexityFitter: Determine empirical Big-O complexity         │
│  - Try: O(1), O(log n), O(n), O(n log n), O(n²), O(n³)         │
│  - Select: Best R² fit (prefer simpler if similar)              │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ (results)
         │
┌────────▼────────────────────────────────────────────────────────┐
│                      Results Layer                              │
│  ProfileResult: Complete profiling data and analysis            │
└─────────┬──────────┬──────────────┬─────────────────────────────┘
          │          │              │
    ┌─────▼──┐  ┌───▼────┐  ┌─────▼──────────┐
    │Formatter│  │Charts  │  │Exporters       │
    │(console)│  │(PNG,   │  │(JSON, CSV)     │
    │         │  │ SVG)   │  │                │
    └─────────┘  └────────┘  └────────────────┘
          │          │              │
          └──────────┴──────────────┘
                    │
                    v
                   User
```

---

## Extension Points

### Adding a New Algorithm

The protocol-based design makes adding new algorithms straightforward.

#### Step 1: Implement the Algorithm Protocol

Create a new class that implements `Algorithm[T]` protocol:

```python
from complexity-profiler.algorithms.base import (
    Algorithm, AlgorithmMetadata, ComplexityClass, MetricsCollector
)

class MyCustomAlgorithm:
    """Your custom algorithm implementation."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Define algorithm metadata."""
        return AlgorithmMetadata(
            name="My Custom Algorithm",
            category="sorting",  # or "searching", "graph", etc.
            expected_complexity=ComplexityClass.LINEARITHMIC,
            space_complexity="O(n)",
            stable=True,
            in_place=False,
            description="Description of your algorithm"
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        """
        Execute your algorithm.

        Args:
            data: Input data to process
            collector: Metrics observer to track operations

        Returns:
            Processed data
        """
        # Your algorithm implementation
        for i in range(len(data)):
            collector.record_access()
            for j in range(i):
                collector.record_comparison()
                if data[j] > data[i]:
                    collector.record_swap()
                    data[j], data[i] = data[i], data[j]

        return data
```

#### Step 2: Register the Algorithm

Register your algorithm in the CLI's algorithm registry:

```python
# In cli/main.py
from complexity-profiler.algorithms.custom import MyCustomAlgorithm

ALGORITHMS = {
    # ... existing algorithms ...
    "my_custom": MyCustomAlgorithm(),
}
```

#### Step 3: Use the Algorithm

Now you can use it with the CLI:

```bash
# Analyze your custom algorithm
complexity-profiler analyze my_custom --sizes 100,500,1000

# Compare with others
complexity-profiler compare my_custom merge_sort quick_sort
```

### Adding a New Metrics Collector

Implement the `MetricsCollector` protocol:

```python
class CustomMetricsCollector:
    """Custom metrics collection implementation."""

    def __init__(self):
        self._metrics = PerformanceMetrics()

    def record_comparison(self) -> None:
        self._metrics.comparisons += 1

    def record_swap(self) -> None:
        self._metrics.swaps += 1

    def record_access(self) -> None:
        self._metrics.accesses += 1

    def record_recursive_call(self) -> None:
        self._metrics.recursive_calls += 1

    def get_metrics(self) -> PerformanceMetrics:
        return self._metrics

    def reset(self) -> None:
        self._metrics = PerformanceMetrics()
```

### Adding New Complexity Classes

Extend the curve fitting to support additional complexity classes:

```python
# In analysis/curve_fitting.py
class ComplexityFitter:
    COMPLEXITY_FUNCTIONS = {
        # ... existing ...
        ComplexityClass.CUSTOM: lambda n, a, b: a * n**(1.5) + b,
    }
```

### Adding New Data Generators

Create generators for specialized test data:

```python
# In data/generators.py
def gaussian_distribution_generator(n: int) -> list[int]:
    """Generate data from Gaussian distribution."""
    return [int(np.random.normal(n/2, n/10)) for _ in range(n)]

def get_data_generator(generator_type: str):
    generators = {
        "random": random_generator,
        "sorted": sorted_generator,
        "reverse": reverse_generator,
        "nearly_sorted": nearly_sorted_generator,
        "gaussian": gaussian_distribution_generator,  # New!
    }
    return generators.get(generator_type, random_generator)
```

### Adding Export Formats

Extend exporters for new formats:

```python
# In export/xml_exporter.py
from xml.etree import ElementTree as ET
from pathlib import Path
from complexity-profiler.analysis.profiler import ProfileResult

def export_to_xml(result: ProfileResult, output_path: Path) -> None:
    """Export profiling results to XML."""
    root = ET.Element("profile")

    algorithm = ET.SubElement(root, "algorithm")
    ET.SubElement(algorithm, "name").text = result.algorithm_name
    ET.SubElement(algorithm, "complexity").text = result.empirical_complexity
    ET.SubElement(algorithm, "r_squared").text = str(result.r_squared)

    # Add metrics data...

    tree = ET.ElementTree(root)
    tree.write(output_path)
```

Then register it in the CLI:

```python
@click.option("--export", type=click.Choice(["json", "csv", "xml"]))
```

---

## Component Interactions

### Profiler-Algorithm Interaction

```python
# Profiler orchestrates, Algorithm executes
profiler = AlgorithmProfiler()
profiler.with_sizes([100, 500, 1000])
profiler.with_runs(5)
profiler.with_data_generator(data_generator)

# Algorithm receives configuration through execution
result = profiler.profile(algorithm)

# Inside profiler.profile():
for size in sizes:
    for run in range(runs):
        collector = DefaultMetricsCollector()
        data_copy = copy.deepcopy(data_generator(size))

        with collector.timing():
            # Algorithm executes with injected collector
            algorithm.execute(data_copy, collector)

        metrics = collector.get_metrics()
        # Metrics are collected and aggregated
```

### Metrics Collection Chain

```
Algorithm
    |
    | calls collector.record_comparison()
    |
    v
MetricsCollector
    |
    | (Polymorphic - could be DefaultMetricsCollector, ThreadSafeMetricsCollector, etc.)
    |
    +-- Increment comparisons counter
    +-- Track execution time
    +-- Accumulate in PerformanceMetrics
    |
    v
PerformanceMetrics
    |
    | returned to profiler
    |
    v
AlgorithmProfiler
    |
    | aggregates across runs per size
    |
    v
statistics.compute_statistics()
    |
    v
curve_fitting.ComplexityFitter.fit_complexity()
    |
    v
ProfileResult
```

### Configuration Flow

```
TOML Config File
    |
    v
config.settings.Settings.load()
    |
    +-- Load general settings
    +-- Load profiling settings
    +-- Load visualization settings
    +-- Load export settings
    +-- Load logging settings
    |
    v (Environment variable overrides applied)
    |
    v
Components receive configuration
    |
    +-- AlgorithmProfiler uses profiling settings
    +-- ChartGenerator uses visualization settings
    +-- Exporters use export settings
    +-- Logging system uses logging settings
```

### CLI-Component Interaction

```
User Input (CLI)
    |
    v
Click Command Handler
    |
    +-- Parse arguments (--sizes, --runs, --data-type)
    +-- Load configuration (Settings)
    +-- Create algorithm instance
    +-- Create profiler
    |
    v
AlgorithmProfiler
    |
    +-- Configure with arguments
    +-- Execute profiling workflow
    +-- Return ProfileResult
    |
    v
Format and Display
    |
    +-- formatters.format_profile_result()
    +-- console.print() (Rich)
    |
    v
Optional: Export
    |
    +-- json_exporter.export_to_json()
    +-- csv_exporter.export_to_csv()
    |
    v
Optional: Visualize
    |
    +-- visualization.charts.ChartGenerator.plot_complexity_curve()
    +-- Display or save chart
```

---

## Type Safety

### Generic Type Parameters

The analyzer uses Python's generic type system extensively:

```python
# Generic algorithm protocol
class Algorithm(Protocol[T]):
    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]: ...

# Generic sorting algorithms
class MergeSort(Generic[T]):
    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]: ...

# Type-safe usage
integers: list[int] = [5, 2, 8, 1, 9]
sorter: Algorithm[int] = MergeSort[int]()
result = sorter.execute(integers, collector)  # Result is list[int]

# Strings work too
words: list[str] = ["zebra", "apple", "banana"]
sorter = MergeSort[str]()
sorted_words = sorter.execute(words, collector)  # Result is list[str]
```

### Type Checking with MyPy

The codebase passes MyPy strict type checking:

```bash
# Verify type safety
mypy complexity-profiler/
```

### Protocol Structural Typing

Protocols enable structural typing without explicit inheritance:

```python
# This class doesn't inherit from Algorithm
class CustomSorter:
    @property
    def metadata(self) -> AlgorithmMetadata: ...

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]: ...

# But it's still recognized as an Algorithm
sorter = CustomSorter()
assert isinstance(sorter, Algorithm)  # True!

profiler.profile(sorter)  # Works without inheritance
```

### Dataclass Type Safety

All data containers use typed dataclasses:

```python
@dataclass
class PerformanceMetrics:
    comparisons: int = 0
    swaps: int = 0
    accesses: int = 0
    recursive_calls: int = 0
    execution_time: float = 0.0
    memory_operations: int = 0
    memory_usage: Optional[int] = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

# Type-safe access
metrics = collector.get_metrics()
total_ops: int = metrics.total_operations()  # Type is known
```

---

## Summary

The Big-O Complexity Analyzer architecture demonstrates modern Python design principles:

1. **Protocols for Abstraction**: Structural typing with runtime checking
2. **Separation of Concerns**: Clear layering from algorithm to UI
3. **Design Patterns**: Strategy, Builder, Observer, and Dependency Injection
4. **Type Safety**: Full generic support and MyPy validation
5. **Extensibility**: Multiple extension points for customization
6. **Testability**: Injected dependencies enable isolated testing
7. **Composition**: Flexible component combinations without inheritance

This design enables the framework to be both professional-grade and developer-friendly, with clear extension points for new algorithms, data generators, and analysis techniques.
