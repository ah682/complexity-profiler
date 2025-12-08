"""
Data generators for algorithm testing and profiling.

This module provides various data generation functions for testing algorithms
under different conditions (random, sorted, reverse-sorted, nearly-sorted, etc.).
These generators are essential for comprehensive algorithm analysis.
"""

from typing import Callable, Optional
import random


def random_data(size: int, seed: Optional[int] = None) -> list[int]:
    """
    Generate a list of random integers.

    Creates a list of the specified size containing random integers
    in the range [0, size * 10).

    Args:
        size: Number of elements to generate
        seed: Random seed for reproducibility (optional)

    Returns:
        List of random integers

    Raises:
        ValueError: If size is not positive

    Example:
        >>> data = random_data(5, seed=42)
        >>> len(data)
        5
        >>> data2 = random_data(5, seed=42)
        >>> data == data2  # Same seed produces same data
        True
    """
    if size <= 0:
        raise ValueError("size must be positive")

    if seed is not None:
        random.seed(seed)

    return [random.randint(0, size * 10) for _ in range(size)]


def sorted_data(size: int) -> list[int]:
    """
    Generate a sorted list of integers.

    Creates a list containing integers from 0 to size-1 in ascending order.
    This represents the best case for many sorting algorithms and worst case
    for others.

    Args:
        size: Number of elements to generate

    Returns:
        Sorted list of integers from 0 to size-1

    Raises:
        ValueError: If size is not positive

    Example:
        >>> sorted_data(5)
        [0, 1, 2, 3, 4]
    """
    if size <= 0:
        raise ValueError("size must be positive")

    return list(range(size))


def reverse_sorted_data(size: int) -> list[int]:
    """
    Generate a reverse-sorted list of integers.

    Creates a list containing integers from size-1 to 0 in descending order.
    This represents the worst case for many sorting algorithms.

    Args:
        size: Number of elements to generate

    Returns:
        Reverse-sorted list of integers from size-1 to 0

    Raises:
        ValueError: If size is not positive

    Example:
        >>> reverse_sorted_data(5)
        [4, 3, 2, 1, 0]
    """
    if size <= 0:
        raise ValueError("size must be positive")

    return list(range(size - 1, -1, -1))


def nearly_sorted_data(size: int, swaps: int = 10, seed: Optional[int] = None) -> list[int]:
    """
    Generate a nearly-sorted list with a few random swaps.

    Creates a sorted list and then performs a specified number of random
    swaps to introduce slight disorder. Useful for testing adaptive algorithms
    and real-world scenarios where data is often partially sorted.

    Args:
        size: Number of elements to generate
        swaps: Number of random swaps to perform (default: 10)
        seed: Random seed for reproducibility (optional)

    Returns:
        Nearly-sorted list of integers

    Raises:
        ValueError: If size is not positive or swaps is negative

    Example:
        >>> data = nearly_sorted_data(10, swaps=2, seed=42)
        >>> len(data)
        10
        >>> # Most elements should still be close to their sorted position
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if swaps < 0:
        raise ValueError("swaps cannot be negative")

    if seed is not None:
        random.seed(seed)

    data = list(range(size))

    # Perform random swaps
    num_swaps = min(swaps, size // 2)  # Cap swaps at half the size
    for _ in range(num_swaps):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        data[i], data[j] = data[j], data[i]

    return data


def duplicates_data(
    size: int,
    num_unique: int = 10,
    seed: Optional[int] = None
) -> list[int]:
    """
    Generate a list with many duplicate values.

    Creates a list where values are chosen from a limited set of unique values,
    resulting in many duplicates. Useful for testing stable sort implementations
    and algorithms that handle duplicates differently.

    Args:
        size: Number of elements to generate
        num_unique: Number of unique values to choose from (default: 10)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of integers with many duplicates

    Raises:
        ValueError: If size or num_unique is not positive

    Example:
        >>> data = duplicates_data(20, num_unique=3, seed=42)
        >>> len(data)
        20
        >>> len(set(data)) <= 3
        True
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if num_unique <= 0:
        raise ValueError("num_unique must be positive")

    if seed is not None:
        random.seed(seed)

    unique_values = list(range(num_unique))
    return [random.choice(unique_values) for _ in range(size)]


def uniform_data(size: int, value: int = 42) -> list[int]:
    """
    Generate a list where all elements are the same.

    Creates a list where every element has the same value. This is an edge case
    useful for testing how algorithms handle completely uniform data.

    Args:
        size: Number of elements to generate
        value: The value to use for all elements (default: 42)

    Returns:
        List with all elements equal to value

    Raises:
        ValueError: If size is not positive

    Example:
        >>> uniform_data(5, value=7)
        [7, 7, 7, 7, 7]
    """
    if size <= 0:
        raise ValueError("size must be positive")

    return [value] * size


def sawtooth_data(size: int, wave_size: int = 10) -> list[int]:
    """
    Generate a sawtooth pattern of integers.

    Creates a list with a repeating ascending pattern (sawtooth wave).
    Useful for testing algorithms on patterned data.

    Args:
        size: Number of elements to generate
        wave_size: Size of each sawtooth wave (default: 10)

    Returns:
        List following a sawtooth pattern

    Raises:
        ValueError: If size or wave_size is not positive

    Example:
        >>> sawtooth_data(15, wave_size=5)
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")

    return [i % wave_size for i in range(size)]


def get_data_generator(name: str) -> Callable[[int], list[int]]:
    """
    Get a data generator function by name.

    Provides a convenient way to retrieve generators using string names,
    useful for configuration-based testing.

    Args:
        name: Name of the generator ('random', 'sorted', 'reverse',
              'nearly_sorted', 'duplicates', 'uniform', 'sawtooth')

    Returns:
        Generator function that takes size and returns a list

    Raises:
        ValueError: If generator name is not recognized

    Example:
        >>> gen = get_data_generator('sorted')
        >>> gen(5)
        [0, 1, 2, 3, 4]
        >>> gen = get_data_generator('random')
        >>> len(gen(10))
        10
    """
    generators = {
        'random': random_data,
        'sorted': sorted_data,
        'reverse': reverse_sorted_data,
        'reverse_sorted': reverse_sorted_data,
        'nearly_sorted': nearly_sorted_data,
        'duplicates': duplicates_data,
        'uniform': uniform_data,
        'sawtooth': sawtooth_data,
    }

    if name not in generators:
        available = ', '.join(sorted(generators.keys()))
        raise ValueError(
            f"Unknown generator '{name}'. Available generators: {available}"
        )

    return generators[name]


# Predefined generator configurations for common use cases
BEST_CASE_GENERATORS = {
    'sorted': sorted_data,
    'nearly_sorted': lambda n: nearly_sorted_data(n, swaps=n // 100),
}

WORST_CASE_GENERATORS = {
    'reverse': reverse_sorted_data,
    'random': random_data,
}

AVERAGE_CASE_GENERATORS = {
    'random': random_data,
    'nearly_sorted': lambda n: nearly_sorted_data(n, swaps=n // 10),
}


__all__ = [
    'random_data',
    'sorted_data',
    'reverse_sorted_data',
    'nearly_sorted_data',
    'duplicates_data',
    'uniform_data',
    'sawtooth_data',
    'get_data_generator',
    'BEST_CASE_GENERATORS',
    'WORST_CASE_GENERATORS',
    'AVERAGE_CASE_GENERATORS',
]
