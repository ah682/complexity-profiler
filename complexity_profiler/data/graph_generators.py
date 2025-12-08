"""
Graph data generators for Big-O Complexity Analyzer.

This module provides utilities for generating various types of graphs for testing
and analyzing graph algorithms. All generators create Graph instances using the
adjacency list representation.

Available Generators:
    - random_graph: Creates a random graph with specified vertices and edge probability
    - complete_graph: Creates a complete graph where every vertex connects to all others
    - tree_graph: Creates a tree structure (connected acyclic graph)
"""

from typing import TypeVar, List
import random
from complexity_profiler.algorithms.base import Comparable
from complexity_profiler.algorithms.graph import Graph


T = TypeVar('T', bound=Comparable)


def random_graph(
    vertices: int,
    edge_probability: float = 0.3,
    directed: bool = False,
    weighted: bool = False,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    seed: int | None = None,
) -> Graph[int]:
    """
    Generate a random graph with specified number of vertices and edge probability.

    Each possible edge is included with the given probability. For undirected graphs,
    this means each pair of vertices has edge_probability chance of being connected.

    Args:
        vertices: Number of vertices in the graph (must be positive)
        edge_probability: Probability (0.0 to 1.0) of an edge existing between any two vertices
        directed: If True, creates a directed graph; if False, undirected
        weighted: If True, assigns random weights to edges
        min_weight: Minimum edge weight (if weighted=True)
        max_weight: Maximum edge weight (if weighted=True)
        seed: Random seed for reproducibility (optional)

    Returns:
        Graph with specified properties

    Raises:
        ValueError: If vertices <= 0 or edge_probability not in [0, 1]

    Example:
        >>> g = random_graph(5, edge_probability=0.5, seed=42)
        >>> print(g.get_vertex_count())
        5
        >>> g = random_graph(10, edge_probability=0.3, weighted=True, min_weight=1.0, max_weight=5.0)
    """
    if vertices <= 0:
        raise ValueError("Number of vertices must be positive")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("Edge probability must be between 0.0 and 1.0")

    if seed is not None:
        random.seed(seed)

    graph = Graph[int](directed=directed)

    # Add all vertices
    for i in range(vertices):
        graph.add_vertex(i)

    # Generate edges based on probability
    for i in range(vertices):
        # For directed graphs, check all vertices
        # For undirected graphs, only check vertices after i to avoid duplicates
        start_j = i + 1 if not directed else 0

        for j in range(start_j, vertices):
            if i == j:  # Skip self-loops
                continue

            # Decide if edge exists based on probability
            if random.random() < edge_probability:
                weight = 1.0
                if weighted:
                    weight = random.uniform(min_weight, max_weight)

                graph.add_edge(i, j, weight=weight)

    return graph


def complete_graph(
    vertices: int,
    directed: bool = False,
    weighted: bool = False,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    seed: int | None = None,
) -> Graph[int]:
    """
    Generate a complete graph where every vertex is connected to every other vertex.

    In a complete graph with n vertices:
    - Undirected: n(n-1)/2 edges
    - Directed: n(n-1) edges

    Args:
        vertices: Number of vertices in the graph (must be positive)
        directed: If True, creates a directed graph; if False, undirected
        weighted: If True, assigns random weights to edges
        min_weight: Minimum edge weight (if weighted=True)
        max_weight: Maximum edge weight (if weighted=True)
        seed: Random seed for reproducibility (optional)

    Returns:
        Complete graph with specified properties

    Raises:
        ValueError: If vertices <= 0

    Example:
        >>> g = complete_graph(5)
        >>> print(g.get_edge_count())
        10
        >>> # For 5 vertices: 5 * 4 / 2 = 10 edges in undirected graph
    """
    if vertices <= 0:
        raise ValueError("Number of vertices must be positive")

    if seed is not None:
        random.seed(seed)

    graph = Graph[int](directed=directed)

    # Add all vertices
    for i in range(vertices):
        graph.add_vertex(i)

    # Connect every vertex to every other vertex
    for i in range(vertices):
        for j in range(vertices):
            if i == j:  # Skip self-loops
                continue

            # For undirected graphs, only add edge once (i < j)
            if not directed and i >= j:
                continue

            weight = 1.0
            if weighted:
                weight = random.uniform(min_weight, max_weight)

            graph.add_edge(i, j, weight=weight)

    return graph


def tree_graph(
    vertices: int,
    branching_factor: int | None = None,
    weighted: bool = False,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    seed: int | None = None,
) -> Graph[int]:
    """
    Generate a tree graph (connected acyclic undirected graph).

    A tree with n vertices has exactly n-1 edges. This generator creates a tree
    by randomly connecting vertices while ensuring no cycles are formed.

    Args:
        vertices: Number of vertices in the tree (must be positive)
        branching_factor: Maximum number of children per node (None = random, 1-3 typically)
        weighted: If True, assigns random weights to edges
        min_weight: Minimum edge weight (if weighted=True)
        max_weight: Maximum edge weight (if weighted=True)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tree graph (undirected, connected, acyclic)

    Raises:
        ValueError: If vertices <= 0

    Example:
        >>> g = tree_graph(10, seed=42)
        >>> print(g.get_vertex_count())
        10
        >>> print(g.get_edge_count())
        9
        >>> # Tree with n vertices always has n-1 edges
    """
    if vertices <= 0:
        raise ValueError("Number of vertices must be positive")

    if seed is not None:
        random.seed(seed)

    graph = Graph[int](directed=False)  # Trees are undirected

    # Single vertex tree
    if vertices == 1:
        graph.add_vertex(0)
        return graph

    # Add all vertices
    for i in range(vertices):
        graph.add_vertex(i)

    # Build tree by connecting each new vertex to an existing one
    # This ensures connectivity and no cycles
    connected_vertices: List[int] = [0]  # Start with vertex 0

    for i in range(1, vertices):
        # Choose a random vertex from already connected vertices to be the parent
        parent = random.choice(connected_vertices)

        weight = 1.0
        if weighted:
            weight = random.uniform(min_weight, max_weight)

        graph.add_edge(parent, i, weight=weight)
        connected_vertices.append(i)

        # If branching factor is specified, limit how many children a node can have
        if branching_factor is not None:
            # Count current children of each vertex
            children_count: dict[int, int] = {}
            for vertex in connected_vertices:
                children_count[vertex] = sum(
                    1 for _, _ in graph.get_neighbors(vertex)
                    if _ > vertex  # Count only children (higher indices)
                )

            # Remove vertices that have reached their branching limit
            connected_vertices = [
                v for v in connected_vertices
                if children_count.get(v, 0) < branching_factor
            ]

            # Ensure at least one vertex is available for connection
            if not connected_vertices:
                connected_vertices = [i]

    return graph


def linear_graph(
    vertices: int,
    weighted: bool = False,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    seed: int | None = None,
) -> Graph[int]:
    """
    Generate a linear graph (path graph) where vertices form a single path.

    A linear graph with n vertices has n-1 edges, forming a straight line:
    0 -- 1 -- 2 -- 3 -- ... -- (n-1)

    Args:
        vertices: Number of vertices in the graph (must be positive)
        weighted: If True, assigns random weights to edges
        min_weight: Minimum edge weight (if weighted=True)
        max_weight: Maximum edge weight (if weighted=True)
        seed: Random seed for reproducibility (optional)

    Returns:
        Linear graph (undirected path)

    Raises:
        ValueError: If vertices <= 0

    Example:
        >>> g = linear_graph(5)
        >>> print(g.get_vertex_count())
        5
        >>> print(g.get_edge_count())
        4
    """
    if vertices <= 0:
        raise ValueError("Number of vertices must be positive")

    if seed is not None:
        random.seed(seed)

    graph = Graph[int](directed=False)

    # Add all vertices
    for i in range(vertices):
        graph.add_vertex(i)

    # Connect vertices in a line
    for i in range(vertices - 1):
        weight = 1.0
        if weighted:
            weight = random.uniform(min_weight, max_weight)

        graph.add_edge(i, i + 1, weight=weight)

    return graph


def cyclic_graph(
    vertices: int,
    weighted: bool = False,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    seed: int | None = None,
) -> Graph[int]:
    """
    Generate a cyclic graph where vertices form a single cycle.

    A cyclic graph with n vertices has n edges, forming a closed loop:
    0 -- 1 -- 2 -- 3 -- ... -- (n-1) -- 0

    Args:
        vertices: Number of vertices in the graph (must be >= 3)
        weighted: If True, assigns random weights to edges
        min_weight: Minimum edge weight (if weighted=True)
        max_weight: Maximum edge weight (if weighted=True)
        seed: Random seed for reproducibility (optional)

    Returns:
        Cyclic graph (undirected cycle)

    Raises:
        ValueError: If vertices < 3

    Example:
        >>> g = cyclic_graph(5)
        >>> print(g.get_vertex_count())
        5
        >>> print(g.get_edge_count())
        5
    """
    if vertices < 3:
        raise ValueError("Cyclic graph must have at least 3 vertices")

    if seed is not None:
        random.seed(seed)

    graph = Graph[int](directed=False)

    # Add all vertices
    for i in range(vertices):
        graph.add_vertex(i)

    # Connect vertices in a cycle
    for i in range(vertices):
        weight = 1.0
        if weighted:
            weight = random.uniform(min_weight, max_weight)

        # Connect i to (i+1) % vertices to form a cycle
        next_vertex = (i + 1) % vertices
        graph.add_edge(i, next_vertex, weight=weight)

    return graph


__all__ = [
    "random_graph",
    "complete_graph",
    "tree_graph",
    "linear_graph",
    "cyclic_graph",
]
