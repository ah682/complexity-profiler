"""
Searching algorithm implementations for Big-O Complexity Analyzer.

This module provides four fundamental searching algorithms, each implemented as a class
following the Algorithm protocol. All algorithms track detailed metrics including
comparisons and array accesses for complexity analysis.

Available Algorithms:
    - LinearSearch: Simple sequential search - O(n)
    - BinarySearch: Efficient search on sorted data - O(log n)
    - JumpSearch: Block-based search on sorted data - O(n)
    - InterpolationSearch: Position-based search - O(log log n) average

Note: Search algorithms expect the target value as the first element of the data list.
"""

from typing import TypeVar, Generic
import math
from complexity_profiler.algorithms.base import (
    AlgorithmMetadata,
    ComplexityClass,
    MetricsCollector,
    Comparable,
)


T = TypeVar('T', bound=Comparable)


class LinearSearch(Generic[T]):
    """
    Linear Search: A simple sequential search algorithm.

    Linear search examines each element in the list sequentially until the target
    element is found or the end of the list is reached. It works on both sorted
    and unsorted data.

    Time Complexity:
        - Best Case: O(1) - target is the first element
        - Average Case: O(n) - target is in the middle
        - Worst Case: O(n) - target is at the end or not present

    Space Complexity: O(1) - no additional space required

    Characteristics:
        - Works on unsorted data
        - Simple to implement
        - No preprocessing required
        - Suitable for small datasets or one-time searches

    Use Cases:
        - Small datasets where overhead of sorting isn't justified
        - Unsorted data
        - When simplicity is valued over performance
        - Linked lists or other non-indexed data structures
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing linear search's properties."""
        return AlgorithmMetadata(
            name="Linear Search",
            category="searching",
            expected_complexity=ComplexityClass.LINEAR,
            space_complexity="O(1)",
            stable=False,
            in_place=True,
            description=(
                "Sequential search that examines each element one by one until "
                "the target is found. Works on both sorted and unsorted data. "
                "Simple but inefficient for large datasets."
            ),
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        """
        Perform linear search on the data.

        The first element of data is treated as the target value to search for.
        Returns a list containing the index where the target was found, or [-1] if not found.

        Args:
            data: List where first element is the target, rest is the search space
            collector: Metrics collector for tracking operations

        Returns:
            List containing [index] where target was found, or [-1] if not found
        """
        if len(data) < 2:
            return [-1]

        target = data[0]
        search_space = data[1:]
        collector.record_access()  # Access target

        index = self._linear_search(search_space, target, collector)
        return [index]

    def _linear_search(
        self, arr: list[T], target: T, collector: MetricsCollector
    ) -> int:
        """
        Internal linear search implementation.

        Args:
            arr: Array to search
            target: Value to find
            collector: Metrics collector

        Returns:
            Index of target in arr, or -1 if not found
        """
        for i, element in enumerate(arr):
            collector.record_access()  # Access element
            collector.record_comparison()  # Compare with target

            if element == target:
                return i

        return -1


class BinarySearch(Generic[T]):
    """
    Binary Search: An efficient divide-and-conquer search algorithm.

    Binary search repeatedly divides the sorted search space in half by comparing
    the target with the middle element. This algorithm requires the data to be
    sorted beforehand.

    Time Complexity:
        - Best Case: O(1) - target is the middle element
        - Average Case: O(log n)
        - Worst Case: O(log n)

    Space Complexity: O(1) - iterative implementation

    Characteristics:
        - Requires sorted data
        - Very efficient for large datasets
        - Logarithmic time complexity
        - Random access required (arrays, not linked lists)

    Use Cases:
        - Large sorted datasets
        - When repeated searches are needed (sort once, search many times)
        - Dictionary lookups
        - Database index searches
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing binary search's properties."""
        return AlgorithmMetadata(
            name="Binary Search",
            category="searching",
            expected_complexity=ComplexityClass.LOGARITHMIC,
            space_complexity="O(1)",
            stable=False,
            in_place=True,
            description=(
                "Efficient divide-and-conquer algorithm that repeatedly halves "
                "the search space by comparing with the middle element. Requires "
                "sorted data. Achieves O(log n) complexity."
            ),
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        """
        Perform binary search on sorted data.

        The first element of data is treated as the target value to search for.
        The remaining elements must be sorted in ascending order.

        Args:
            data: List where first element is the target, rest is the sorted search space
            collector: Metrics collector for tracking operations

        Returns:
            List containing [index] where target was found, or [-1] if not found
        """
        if len(data) < 2:
            return [-1]

        target = data[0]
        search_space = data[1:]
        collector.record_access()  # Access target

        index = self._binary_search(search_space, target, collector)
        return [index]

    def _binary_search(
        self, arr: list[T], target: T, collector: MetricsCollector
    ) -> int:
        """
        Internal iterative binary search implementation.

        Args:
            arr: Sorted array to search
            target: Value to find
            collector: Metrics collector

        Returns:
            Index of target in arr, or -1 if not found
        """
        left, right = 0, len(arr) - 1

        while left <= right:
            collector.record_comparison()  # Loop condition

            # Calculate middle index (avoid overflow)
            mid = left + (right - left) // 2
            collector.record_access()  # Access middle element

            collector.record_comparison()
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                collector.record_comparison()
                left = mid + 1
            else:
                right = mid - 1

        return -1


class JumpSearch(Generic[T]):
    """
    Jump Search: A block-based search algorithm for sorted data.

    Jump search improves upon linear search by jumping ahead by fixed steps
    (block size) instead of searching sequentially. Once the block containing
    the target is found, linear search is performed within that block. The
    optimal jump size is n.

    Time Complexity:
        - Best Case: O(1) - target is at a jump position
        - Average Case: O(n)
        - Worst Case: O(n)

    Space Complexity: O(1)

    Characteristics:
        - Requires sorted data
        - Better than linear search, worse than binary search
        - Requires fewer comparisons than linear for large n
        - Can be better than binary search when jumping back is costly

    Use Cases:
        - Sorted arrays where binary search is not ideal
        - Systems where jumping backward is expensive
        - When n comparisons are acceptable
        - Combination with other search strategies
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing jump search's properties."""
        return AlgorithmMetadata(
            name="Jump Search",
            category="searching",
            expected_complexity=ComplexityClass.LINEAR,  # Using LINEAR as closest to O(n)
            space_complexity="O(1)",
            stable=False,
            in_place=True,
            description=(
                "Block-based search that jumps ahead by n steps to find the block "
                "containing the target, then performs linear search within that block. "
                "Achieves O(n) complexity on sorted data."
            ),
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        """
        Perform jump search on sorted data.

        The first element of data is treated as the target value to search for.
        The remaining elements must be sorted in ascending order.

        Args:
            data: List where first element is the target, rest is the sorted search space
            collector: Metrics collector for tracking operations

        Returns:
            List containing [index] where target was found, or [-1] if not found
        """
        if len(data) < 2:
            return [-1]

        target = data[0]
        search_space = data[1:]
        collector.record_access()  # Access target

        index = self._jump_search(search_space, target, collector)
        return [index]

    def _jump_search(
        self, arr: list[T], target: T, collector: MetricsCollector
    ) -> int:
        """
        Internal jump search implementation.

        Args:
            arr: Sorted array to search
            target: Value to find
            collector: Metrics collector

        Returns:
            Index of target in arr, or -1 if not found
        """
        n = len(arr)
        if n == 0:
            return -1

        # Calculate optimal jump size: n
        step = int(math.sqrt(n))
        prev = 0

        # Find the block where target may exist
        while prev < n:
            collector.record_access()
            collector.record_comparison()

            # Check if we've found the block
            current = min(prev + step, n) - 1
            collector.record_access()

            if arr[current] >= target:
                break

            prev += step

        # Linear search within the block
        for i in range(prev, min(prev + step, n)):
            collector.record_access()
            collector.record_comparison()

            if arr[i] == target:
                return i
            elif arr[i] > target:
                collector.record_comparison()
                break

        return -1


class InterpolationSearch(Generic[T]):
    """
    Interpolation Search: A position-based search for uniformly distributed sorted data.

    Interpolation search improves upon binary search by calculating the probable
    position of the target based on its value, rather than always checking the
    middle. It works best on uniformly distributed data.

    Time Complexity:
        - Best Case: O(1) - target found immediately
        - Average Case: O(log log n) - for uniformly distributed data
        - Worst Case: O(n) - for non-uniform data

    Space Complexity: O(1)

    Characteristics:
        - Requires sorted data
        - Works best on uniformly distributed data
        - Can be faster than binary search for large, uniform datasets
        - Degrades to O(n) on non-uniform data

    Use Cases:
        - Large sorted datasets with uniform distribution
        - Phone books, dictionaries (alphabetically uniform)
        - Numerical data with even distribution
        - When data distribution is known to be uniform

    Note: This implementation works with numeric data. For proper interpolation,
    the type T should support arithmetic operations.
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing interpolation search's properties."""
        return AlgorithmMetadata(
            name="Interpolation Search",
            category="searching",
            expected_complexity=ComplexityClass.LOGARITHMIC,  # O(log log n) average
            space_complexity="O(1)",
            stable=False,
            in_place=True,
            description=(
                "Position-based search that estimates the target's position using "
                "interpolation formula. Achieves O(log log n) on uniformly distributed "
                "sorted data, but can degrade to O(n) on non-uniform data."
            ),
        )

    def execute(self, data: list[T], collector: MetricsCollector) -> list[T]:
        """
        Perform interpolation search on sorted uniformly distributed data.

        The first element of data is treated as the target value to search for.
        The remaining elements must be sorted in ascending order and ideally
        uniformly distributed for optimal performance.

        Args:
            data: List where first element is the target, rest is the sorted search space
            collector: Metrics collector for tracking operations

        Returns:
            List containing [index] where target was found, or [-1] if not found
        """
        if len(data) < 2:
            return [-1]

        target = data[0]
        search_space = data[1:]
        collector.record_access()  # Access target

        index = self._interpolation_search(search_space, target, collector)
        return [index]

    def _interpolation_search(
        self, arr: list[T], target: T, collector: MetricsCollector
    ) -> int:
        """
        Internal interpolation search implementation.

        Args:
            arr: Sorted array with uniform distribution
            target: Value to find
            collector: Metrics collector

        Returns:
            Index of target in arr, or -1 if not found
        """
        if not arr:
            return -1

        low, high = 0, len(arr) - 1

        while low <= high:
            collector.record_comparison()  # Loop condition
            collector.record_access()  # Access arr[low]
            collector.record_access()  # Access arr[high]

            # Check if target is in range
            if arr[low] > target or arr[high] < target:
                collector.record_comparison()
                collector.record_comparison()
                break

            # If the range has converged to one element
            if low == high:
                collector.record_comparison()
                collector.record_access()
                if arr[low] == target:
                    collector.record_comparison()
                    return low
                break

            # Calculate interpolated position
            # For types that don't support arithmetic, this will use comparison-based estimation
            try:
                # Try numeric interpolation (works for int, float)
                pos = low + int(
                    (high - low) * (float(target) - float(arr[low])) /
                    (float(arr[high]) - float(arr[low]))
                )
                # Ensure pos is within bounds
                pos = max(low, min(high, pos))
            except (TypeError, ValueError, ZeroDivisionError):
                # Fall back to binary search behavior if interpolation fails
                pos = low + (high - low) // 2

            collector.record_access()  # Access arr[pos]
            collector.record_comparison()

            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                collector.record_comparison()
                low = pos + 1
            else:
                high = pos - 1

        return -1


__all__ = [
    "LinearSearch",
    "BinarySearch",
    "JumpSearch",
    "InterpolationSearch",
]
