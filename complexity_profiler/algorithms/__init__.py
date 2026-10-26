"""Algorithm implementations for complexity analysis."""

from complexity_profiler.algorithms.base import (
    Algorithm,
    AlgorithmMetadata,
    Comparable,
    ComplexityClass,
    MetricsCollector,
)
from complexity_profiler.algorithms.graph import (
    BFS,
    DFS,
    DijkstraShortestPath,
    Graph,
)
from complexity_profiler.algorithms.searching import (
    BinarySearch,
    InterpolationSearch,
    JumpSearch,
    LinearSearch,
)
from complexity_profiler.algorithms.sorting import (
    BubbleSort,
    HeapSort,
    InsertionSort,
    MergeSort,
    QuickSort,
    SelectionSort,
)

__all__ = [
    # Base protocols and types
    "Algorithm",
    "AlgorithmMetadata",
    "ComplexityClass",
    "Comparable",
    "MetricsCollector",
    # Sorting algorithms
    "MergeSort",
    "QuickSort",
    "SelectionSort",
    "BubbleSort",
    "InsertionSort",
    "HeapSort",
    # Searching algorithms
    "LinearSearch",
    "BinarySearch",
    "JumpSearch",
    "InterpolationSearch",
    # Graph algorithms and structures
    "Graph",
    "BFS",
    "DFS",
    "DijkstraShortestPath",
]
