"""Core protocols and base classes for algorithm implementations."""

from typing import Protocol, TypeVar, Any, runtime_checkable
from dataclasses import dataclass
from enum import Enum


T = TypeVar('T')


class ComplexityClass(Enum):
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n�)"
    CUBIC = "O(n�)"
    EXPONENTIAL = "O(2)"
    FACTORIAL = "O(n!)"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AlgorithmMetadata:
    """Algorithm metadata and properties."""
    name: str
    category: str
    expected_complexity: ComplexityClass
    space_complexity: str
    stable: bool = False
    in_place: bool = False
    description: str = ""


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...


class MetricsCollector(Protocol):
    """Metrics collection during algorithm execution."""
    def record_comparison(self) -> None: ...
    def record_swap(self) -> None: ...
    def record_access(self) -> None: ...
    def record_recursive_call(self) -> None: ...
    def get_metrics(self) -> Any: ...
    def reset(self) -> None: ...


class Algorithm(Protocol[T]):
    """Protocol for algorithm implementations."""

    @property
    def metadata(self) -> AlgorithmMetadata: ...

    def execute(self, data: list[T], collector: 'MetricsCollector') -> list[T]: ...


__all__ = [
    "ComplexityClass",
    "AlgorithmMetadata",
    "Comparable",
    "MetricsCollector",
    "Algorithm",
]
