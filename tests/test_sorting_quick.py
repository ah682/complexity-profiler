"""Quick test to verify sorting algorithms work correctly."""

from complexity_profiler.algorithms.sorting import (
    MergeSort,
    QuickSort,
    SelectionSort,
    BubbleSort,
    InsertionSort,
    HeapSort,
)
from complexity_profiler.algorithms.base import MetricsCollector


class SimpleCollector:
    """Simple metrics collector for testing."""

    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.accesses = 0
        self.recursive_calls = 0

    def record_comparison(self):
        self.comparisons += 1

    def record_swap(self):
        self.swaps += 1

    def record_access(self):
        self.accesses += 1

    def record_recursive_call(self):
        self.recursive_calls += 1

    def get_metrics(self):
        return {
            "comparisons": self.comparisons,
            "swaps": self.swaps,
            "accesses": self.accesses,
            "recursive_calls": self.recursive_calls,
        }

    def reset(self):
        self.comparisons = 0
        self.swaps = 0
        self.accesses = 0
        self.recursive_calls = 0


def test_algorithm(algorithm_class, name: str, test_data: list):
    """Test a sorting algorithm."""
    collector = SimpleCollector()
    algorithm = algorithm_class()

    # Test sorting
    sorted_data = algorithm.execute(test_data.copy(), collector)

    # Verify correctness
    expected = sorted(test_data)
    is_correct = sorted_data == expected

    # Print results
    print(f"\n{name}:")
    print(f"  Metadata: {algorithm.metadata.name}")
    print(f"  Complexity: {algorithm.metadata.expected_complexity}")
    print(f"  Stable: {algorithm.metadata.stable}")
    print(f"  In-place: {algorithm.metadata.in_place}")
    print(f"  Correct: {is_correct}")
    print(f"  Metrics: {collector.get_metrics()}")

    return is_correct


if __name__ == "__main__":
    # Test data
    test_data = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50, 23, 36, 18, 77]

    print(f"Testing with data: {test_data}")
    print(f"Expected result: {sorted(test_data)}")

    algorithms = [
        (MergeSort, "MergeSort"),
        (QuickSort, "QuickSort"),
        (SelectionSort, "SelectionSort"),
        (BubbleSort, "BubbleSort"),
        (InsertionSort, "InsertionSort"),
        (HeapSort, "HeapSort"),
    ]

    all_correct = True
    for algo_class, name in algorithms:
        if not test_algorithm(algo_class, name, test_data):
            all_correct = False

    print("\n" + "=" * 60)
    if all_correct:
        print("✓ All algorithms passed correctness tests!")
    else:
        print("✗ Some algorithms failed!")
