# Custom experiments

Use the Python API when you need control over input generation, seeds, warmups
or result inspection.

## Repeated profiles

```python
from complexity_profiler import AlgorithmProfiler
from complexity_profiler.algorithms.sorting import MergeSort
from complexity_profiler.data.generators import random_data

result = (
    AlgorithmProfiler(warmup_runs=2, min_r_squared=0.8)
    .with_sizes([100, 500, 1000, 5000])
    .with_runs(10)
    .with_data_generator(lambda size: random_data(size, seed=42))
    .profile(MergeSort())
)

print(result.empirical_complexity, result.r_squared)
print(result.candidate_scores)
print(result.notes)
for size in result.input_sizes:
    stats = result.statistics_per_size[size]
    counts = [run.comparisons for run in result.metrics_per_size[size]]
    print(size, stats.mean, stats.std_dev, counts)
```

This generator repeats the same seeded input at a given size, which is useful
for studying timing variation on fixed data. To study variation across inputs,
provide a generator that advances its random state across calls.

## Search inputs

Search algorithms expect `[target, *search_space]` and return the target's index
within the search space (or `-1`) in a one-element list. Binary, jump and
interpolation search need a sorted search space.

```python
from complexity_profiler.algorithms.searching import BinarySearch
from complexity_profiler.analysis.metrics import DefaultMetricsCollector

collector = DefaultMetricsCollector()
indices = BinarySearch().execute([7, 1, 3, 5, 7, 9], collector)
print(indices)  # [3]
print(collector.get_metrics().comparisons)
```

The generic CLI generator puts its first generated value in the target position.
For an average-case experiment, choose successful and unsuccessful targets
explicitly and account for the target element when reporting search-space size.

Graph interfaces and generators are covered in the [algorithm reference](../algorithms.md).
For direct fitting of controlled data, see the [README example](../../README.md#use-as-a-library).
