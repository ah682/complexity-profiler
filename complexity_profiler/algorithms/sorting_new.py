"""Sorting algorithm implementations."""

from typing import TypeVar, Generic
import random
from complexity_profiler.algorithms.base import (
    AlgorithmMetadata,
    ComplexityClass,
    MetricsCollector,
    Comparable,
)


T = TypeVar('T', bound=Comparable)


class MergeSort(Generic[T]):
    """Stable divide-and-conquer sorting algorithm - O(n log n)."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Merge Sort",
            category="sorting",
            expected_complexity=ComplexityClass.LINEARITHMIC,
            space_complexity="O(n)",
            stable=True,
            in_place=False,
            description="Stable divide-and-conquer algorithm with O(n log n) guaranteed performance.",
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        return self._merge_sort(data.copy(), collector)

    def _merge_sort(self, arr: list[T], collector: MetricsCollector) -> list[T]:
        if len(arr) <= 1:
            return arr

        collector.record_recursive_call()

        mid = len(arr) // 2
        collector.record_access()

        left = self._merge_sort(arr[:mid], collector)
        right = self._merge_sort(arr[mid:], collector)

        return self._merge(left, right, collector)

    def _merge(self, left: list[T], right: list[T], collector: MetricsCollector) -> list[T]:
        merged: list[T] = []
        i, j = 0, 0

        while i < len(left) and j < len(right):
            collector.record_comparison()
            collector.record_access()
            collector.record_access()

            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

            collector.record_swap()

        while i < len(left):
            merged.append(left[i])
            collector.record_access()
            collector.record_swap()
            i += 1

        while j < len(right):
            merged.append(right[j])
            collector.record_access()
            collector.record_swap()
            j += 1

        return merged


class QuickSort(Generic[T]):
    """Fast partitioning algorithm - O(n log n) average case."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Quick Sort",
            category="sorting",
            expected_complexity=ComplexityClass.LINEARITHMIC,
            space_complexity="O(log n) to O(n)",
            stable=False,
            in_place=False,
            description="Fast partitioning with random pivot selection for O(n log n) average case.",
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        return self._quick_sort(data.copy(), collector)

    def _quick_sort(self, arr: list[T], collector: MetricsCollector) -> list[T]:
        if len(arr) <= 1:
            return arr

        collector.record_recursive_call()

        pivot = random.choice(arr)
        collector.record_access()

        lesser: list[T] = []
        equal: list[T] = []
        greater: list[T] = []

        for element in arr:
            collector.record_access()
            collector.record_comparison()

            if element < pivot:
                lesser.append(element)
                collector.record_swap()
            elif element == pivot:
                collector.record_comparison()
                equal.append(element)
                collector.record_swap()
            else:
                greater.append(element)
                collector.record_swap()

        sorted_lesser = self._quick_sort(lesser, collector)
        sorted_greater = self._quick_sort(greater, collector)

        return sorted_lesser + equal + sorted_greater


class SelectionSort(Generic[T]):
    """Simple in-place sorting - O(n²)."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Selection Sort",
            category="sorting",
            expected_complexity=ComplexityClass.QUADRATIC,
            space_complexity="O(1)",
            stable=False,
            in_place=True,
            description="Simple in-place algorithm that repeatedly selects the minimum element.",
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        arr = data.copy()
        n = len(arr)

        for i in range(n):
            min_index = i

            for j in range(i + 1, n):
                collector.record_comparison()
                collector.record_access()
                collector.record_access()

                if arr[j] < arr[min_index]:
                    min_index = j

            if min_index != i:
                arr[i], arr[min_index] = arr[min_index], arr[i]
                collector.record_swap()
                collector.record_access()
                collector.record_access()

        return arr


class BubbleSort(Generic[T]):
    """Stable comparison-based sorting - O(n²)."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Bubble Sort",
            category="sorting",
            expected_complexity=ComplexityClass.QUADRATIC,
            space_complexity="O(1)",
            stable=True,
            in_place=True,
            description="Stable in-place algorithm with early termination optimization.",
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        arr = data.copy()
        n = len(arr)

        for i in range(n):
            swapped = False

            for j in range(0, n - i - 1):
                collector.record_comparison()
                collector.record_access()
                collector.record_access()

                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    collector.record_swap()
                    collector.record_access()
                    collector.record_access()
                    swapped = True

            if not swapped:
                break

        return arr


class InsertionSort(Generic[T]):
    """Adaptive stable sorting - O(n²)."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Insertion Sort",
            category="sorting",
            expected_complexity=ComplexityClass.QUADRATIC,
            space_complexity="O(1)",
            stable=True,
            in_place=True,
            description="Stable, adaptive algorithm efficient for small or nearly sorted datasets.",
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        arr = data.copy()
        n = len(arr)

        for i in range(1, n):
            key = arr[i]
            collector.record_access()
            j = i - 1

            while j >= 0:
                collector.record_comparison()
                collector.record_access()

                if arr[j] > key:
                    arr[j + 1] = arr[j]
                    collector.record_swap()
                    collector.record_access()
                    j -= 1
                else:
                    break

            arr[j + 1] = key
            collector.record_swap()
            collector.record_access()

        return arr


class HeapSort(Generic[T]):
    """In-place heap-based sorting - O(n log n)."""

    @property
    def metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Heap Sort",
            category="sorting",
            expected_complexity=ComplexityClass.LINEARITHMIC,
            space_complexity="O(1)",
            stable=False,
            in_place=True,
            description="In-place algorithm using binary heap for guaranteed O(n log n) performance.",
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        arr = data.copy()
        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i, collector)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            collector.record_swap()
            collector.record_access()
            collector.record_access()

            self._heapify(arr, i, 0, collector)

        return arr

    def _heapify(
        self, arr: list[T], heap_size: int, root_index: int, collector: MetricsCollector
    ) -> None:
        largest = root_index
        left_child = 2 * root_index + 1
        right_child = 2 * root_index + 2

        if left_child < heap_size:
            collector.record_comparison()
            collector.record_access()
            collector.record_access()

            if arr[left_child] > arr[largest]:
                largest = left_child

        if right_child < heap_size:
            collector.record_comparison()
            collector.record_access()
            collector.record_access()

            if arr[right_child] > arr[largest]:
                largest = right_child

        if largest != root_index:
            arr[root_index], arr[largest] = arr[largest], arr[root_index]
            collector.record_swap()
            collector.record_access()
            collector.record_access()

            collector.record_recursive_call()
            self._heapify(arr, heap_size, largest, collector)


__all__ = [
    "MergeSort",
    "QuickSort",
    "SelectionSort",
    "BubbleSort",
    "InsertionSort",
    "HeapSort",
]
