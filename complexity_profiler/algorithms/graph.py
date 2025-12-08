"""
Graph algorithm implementations for Big-O Complexity Analyzer.

This module provides fundamental graph algorithms and a Graph data structure
implementation. All algorithms follow the Algorithm protocol and track detailed
metrics for complexity analysis.

Available Algorithms:
    - BFS: Breadth-First Search - O(V + E)
    - DFS: Depth-First Search - O(V + E)
    - DijkstraShortestPath: Single-source shortest path - O((V + E) log V)

Graph Structure:
    - Graph: Adjacency list representation supporting directed/undirected graphs
"""

from typing import TypeVar, Generic, Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq
from complexity_profiler.algorithms.base import (
    AlgorithmMetadata,
    ComplexityClass,
    MetricsCollector,
    Comparable,
)


T = TypeVar('T', bound=Comparable)


@dataclass
class Graph(Generic[T]):
    """
    Graph data structure using adjacency list representation.

    This implementation supports both directed and undirected graphs, with optional
    edge weights for weighted graph algorithms.

    Attributes:
        vertices: Set of all vertices in the graph
        adjacency_list: Dictionary mapping each vertex to list of (neighbor, weight) tuples
        directed: Whether the graph is directed (True) or undirected (False)

    Example:
        >>> g = Graph[int]()
        >>> g.add_edge(1, 2, weight=5.0)
        >>> g.add_edge(2, 3, weight=3.0)
        >>> print(g.get_neighbors(1))
        [(2, 5.0)]
    """

    vertices: Set[T] = field(default_factory=set)
    adjacency_list: Dict[T, List[Tuple[T, float]]] = field(default_factory=dict)
    directed: bool = False

    def add_vertex(self, vertex: T) -> None:
        """
        Add a vertex to the graph.

        Args:
            vertex: The vertex to add
        """
        if vertex not in self.vertices:
            self.vertices.add(vertex)
            self.adjacency_list[vertex] = []

    def add_edge(self, from_vertex: T, to_vertex: T, weight: float = 1.0) -> None:
        """
        Add an edge to the graph.

        For undirected graphs, this adds edges in both directions.

        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            weight: Edge weight (default: 1.0)
        """
        # Ensure vertices exist
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)

        # Add edge
        self.adjacency_list[from_vertex].append((to_vertex, weight))

        # For undirected graphs, add reverse edge
        if not self.directed:
            self.adjacency_list[to_vertex].append((from_vertex, weight))

    def get_neighbors(self, vertex: T) -> List[Tuple[T, float]]:
        """
        Get neighbors of a vertex.

        Args:
            vertex: The vertex to get neighbors for

        Returns:
            List of (neighbor, weight) tuples
        """
        return self.adjacency_list.get(vertex, [])

    def get_vertex_count(self) -> int:
        """
        Get the number of vertices in the graph.

        Returns:
            Number of vertices
        """
        return len(self.vertices)

    def get_edge_count(self) -> int:
        """
        Get the number of edges in the graph.

        For undirected graphs, each edge is counted once.

        Returns:
            Number of edges
        """
        total = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        return total // 2 if not self.directed else total

    def __repr__(self) -> str:
        """String representation of the graph."""
        return (
            f"Graph(vertices={len(self.vertices)}, "
            f"edges={self.get_edge_count()}, "
            f"directed={self.directed})"
        )


class BFS(Generic[T]):
    """
    Breadth-First Search: A graph traversal algorithm.

    BFS explores the graph level by level, visiting all neighbors of a vertex
    before moving to the next level. It uses a queue to maintain the traversal
    order and is useful for finding shortest paths in unweighted graphs.

    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) for the queue and visited set

    Characteristics:
        - Explores graph level by level
        - Finds shortest path in unweighted graphs
        - Uses queue (FIFO) data structure
        - Guarantees minimum number of edges to reach any vertex

    Use Cases:
        - Shortest path in unweighted graphs
        - Level-order traversal
        - Finding connected components
        - Web crawling
        - Social network analysis (friends at distance k)
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing BFS properties."""
        return AlgorithmMetadata(
            name="Breadth-First Search (BFS)",
            category="graph",
            expected_complexity=ComplexityClass.LINEAR,  # O(V + E)
            space_complexity="O(V)",
            stable=False,
            in_place=False,
            description=(
                "Level-by-level graph traversal using a queue. Explores all neighbors "
                "at current depth before moving to next level. Finds shortest path in "
                "unweighted graphs. Time complexity O(V + E)."
            ),
        )

    def execute(self, data: list[Graph[T]], collector: MetricsCollector) -> list[T]:
        """
        Perform BFS traversal on a graph.

        Args:
            data: List containing a single Graph object
            collector: Metrics collector for tracking operations

        Returns:
            List of vertices in BFS traversal order
        """
        if not data or not isinstance(data[0], Graph):
            return []

        graph = data[0]
        collector.record_access()

        if not graph.vertices:
            return []

        # Start from an arbitrary vertex
        start_vertex = next(iter(graph.vertices))
        collector.record_access()

        return self._bfs_traversal(graph, start_vertex, collector)

    def _bfs_traversal(
        self, graph: Graph[T], start: T, collector: MetricsCollector
    ) -> list[T]:
        """
        Internal BFS implementation.

        Args:
            graph: The graph to traverse
            start: Starting vertex
            collector: Metrics collector

        Returns:
            List of vertices in BFS order
        """
        visited: Set[T] = set()
        queue: deque[T] = deque([start])
        result: list[T] = []

        visited.add(start)
        collector.record_access()  # Mark as visited

        while queue:
            collector.record_comparison()  # Queue empty check
            vertex = queue.popleft()
            result.append(vertex)
            collector.record_access()  # Dequeue operation

            # Visit all neighbors
            neighbors = graph.get_neighbors(vertex)
            collector.record_access()

            for neighbor, weight in neighbors:
                collector.record_access()  # Access neighbor
                collector.record_comparison()  # Check if visited

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    collector.record_access()  # Enqueue operation
                    collector.record_swap()  # Count queue append as write

        return result


class DFS(Generic[T]):
    """
    Depth-First Search: A graph traversal algorithm.

    DFS explores as far as possible along each branch before backtracking.
    It uses a stack (or recursion) to maintain the traversal order and is
    useful for topological sorting, cycle detection, and pathfinding.

    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) for the recursion stack and visited set

    Characteristics:
        - Explores deeply before backtracking
        - Uses stack (LIFO) or recursion
        - Good for detecting cycles
        - Natural for tree-like structures

    Use Cases:
        - Topological sorting
        - Cycle detection
        - Pathfinding in mazes
        - Solving puzzles (sudoku, N-queens)
        - Finding strongly connected components
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing DFS properties."""
        return AlgorithmMetadata(
            name="Depth-First Search (DFS)",
            category="graph",
            expected_complexity=ComplexityClass.LINEAR,  # O(V + E)
            space_complexity="O(V)",
            stable=False,
            in_place=False,
            description=(
                "Deep exploration graph traversal using recursion or stack. Explores "
                "as far as possible along each branch before backtracking. Useful for "
                "topological sorting and cycle detection. Time complexity O(V + E)."
            ),
        )

    def execute(self, data: list[Graph[T]], collector: MetricsCollector) -> list[T]:
        """
        Perform DFS traversal on a graph.

        Args:
            data: List containing a single Graph object
            collector: Metrics collector for tracking operations

        Returns:
            List of vertices in DFS traversal order
        """
        if not data or not isinstance(data[0], Graph):
            return []

        graph = data[0]
        collector.record_access()

        if not graph.vertices:
            return []

        # Start from an arbitrary vertex
        start_vertex = next(iter(graph.vertices))
        collector.record_access()

        visited: Set[T] = set()
        result: list[T] = []

        self._dfs_recursive(graph, start_vertex, visited, result, collector)

        return result

    def _dfs_recursive(
        self,
        graph: Graph[T],
        vertex: T,
        visited: Set[T],
        result: list[T],
        collector: MetricsCollector,
    ) -> None:
        """
        Internal recursive DFS implementation.

        Args:
            graph: The graph to traverse
            vertex: Current vertex
            visited: Set of visited vertices
            result: List to store traversal order
            collector: Metrics collector
        """
        collector.record_recursive_call()
        visited.add(vertex)
        result.append(vertex)
        collector.record_access()  # Mark as visited

        # Visit all neighbors
        neighbors = graph.get_neighbors(vertex)
        collector.record_access()

        for neighbor, weight in neighbors:
            collector.record_access()  # Access neighbor
            collector.record_comparison()  # Check if visited

            if neighbor not in visited:
                self._dfs_recursive(graph, neighbor, visited, result, collector)


class DijkstraShortestPath(Generic[T]):
    """
    Dijkstra's Shortest Path Algorithm: Single-source shortest path.

    Dijkstra's algorithm finds the shortest paths from a source vertex to all
    other vertices in a weighted graph with non-negative edge weights. It uses
    a priority queue to greedily select the vertex with minimum distance.

    Time Complexity: O((V + E) log V) with binary heap priority queue
    Space Complexity: O(V) for distances and priority queue

    Characteristics:
        - Works on weighted graphs with non-negative weights
        - Finds shortest path to all reachable vertices
        - Uses greedy approach with priority queue
        - Optimal for single-source shortest path

    Use Cases:
        - GPS navigation systems
        - Network routing protocols
        - Social network analysis (degrees of separation)
        - Game AI pathfinding
        - Resource allocation optimization

    Note: Does not work with negative edge weights (use Bellman-Ford instead)
    """

    @property
    def metadata(self) -> AlgorithmMetadata:
        """Return metadata describing Dijkstra's algorithm properties."""
        return AlgorithmMetadata(
            name="Dijkstra's Shortest Path",
            category="graph",
            expected_complexity=ComplexityClass.LINEARITHMIC,  # O((V + E) log V)
            space_complexity="O(V)",
            stable=False,
            in_place=False,
            description=(
                "Single-source shortest path algorithm using greedy approach with "
                "priority queue. Finds shortest paths to all vertices from source. "
                "Requires non-negative edge weights. Time complexity O((V + E) log V)."
            ),
        )

    def execute(
        self, data: list[Graph[T]], collector: MetricsCollector
    ) -> list[Tuple[T, float]]:
        """
        Find shortest paths from source vertex to all other vertices.

        Args:
            data: List containing a single Graph object
            collector: Metrics collector for tracking operations

        Returns:
            List of (vertex, distance) tuples for all reachable vertices
        """
        if not data or not isinstance(data[0], Graph):
            return []

        graph = data[0]
        collector.record_access()

        if not graph.vertices:
            return []

        # Start from an arbitrary vertex
        source = next(iter(graph.vertices))
        collector.record_access()

        return self._dijkstra(graph, source, collector)

    def _dijkstra(
        self, graph: Graph[T], source: T, collector: MetricsCollector
    ) -> list[Tuple[T, float]]:
        """
        Internal Dijkstra's algorithm implementation.

        Args:
            graph: The weighted graph
            source: Source vertex
            collector: Metrics collector

        Returns:
            List of (vertex, distance) tuples
        """
        # Initialize distances
        distances: Dict[T, float] = {vertex: float('inf') for vertex in graph.vertices}
        distances[source] = 0.0
        collector.record_access()

        # Priority queue: (distance, vertex)
        pq: List[Tuple[float, T]] = [(0.0, source)]
        visited: Set[T] = set()

        while pq:
            collector.record_comparison()  # Queue empty check

            # Get vertex with minimum distance
            current_dist, current_vertex = heapq.heappop(pq)
            collector.record_access()  # Heap pop
            collector.record_comparison()

            # Skip if already processed
            if current_vertex in visited:
                collector.record_comparison()
                continue

            visited.add(current_vertex)
            collector.record_access()

            # Check all neighbors
            neighbors = graph.get_neighbors(current_vertex)
            collector.record_access()

            for neighbor, weight in neighbors:
                collector.record_access()  # Access neighbor and weight
                collector.record_comparison()  # Check if visited

                if neighbor in visited:
                    continue

                # Calculate new distance
                new_distance = current_dist + weight
                collector.record_comparison()

                # Update if shorter path found
                if new_distance < distances[neighbor]:
                    collector.record_comparison()
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))
                    collector.record_access()  # Heap push
                    collector.record_swap()  # Count heap operation

        # Return results as list of (vertex, distance) tuples
        result = [
            (vertex, dist)
            for vertex, dist in distances.items()
            if dist != float('inf')
        ]
        return result


__all__ = [
    "Graph",
    "BFS",
    "DFS",
    "DijkstraShortestPath",
]
