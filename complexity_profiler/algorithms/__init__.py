"""Algorithm implementations for complexity analysis."""

from complexity_profiler.algorithms.base import (
    Algorithm,
    AlgorithmMetadata,
    ComplexityClass,
    Comparable,
    MetricsCollector,
)
from complexity_profiler.algorithms.sorting import (
    MergeSort,
    QuickSort,
    SelectionSort,
    BubbleSort,
    InsertionSort,
    HeapSort,
)
from complexity_profiler.algorithms.searching import (
    LinearSearch,
    BinarySearch,
    JumpSearch,
    InterpolationSearch,
)
from complexity_profiler.algorithms.graph import (
    Graph,
    BFS,
    DFS,
    DijkstraShortestPath,
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
