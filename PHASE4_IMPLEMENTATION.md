# Phase 4 Implementation: Algorithm Expansion

This document describes the Phase 4 implementation for the Big-O Complexity Analyzer, which adds searching algorithms, graph algorithms, and graph data generators.

## Overview

Phase 4 expands the algorithm library with:
- **4 Searching Algorithms**: Linear Search, Binary Search, Jump Search, Interpolation Search
- **3 Graph Algorithms**: BFS, DFS, Dijkstra's Shortest Path
- **5 Graph Generators**: Random, Complete, Tree, Linear, Cyclic graphs

All implementations follow the `Algorithm` protocol and integrate seamlessly with the existing metrics collection system.

## File Structure

```
complexity-profiler/
├── algorithms/
│   ├── searching.py          # NEW: 4 searching algorithms
│   ├── graph.py              # NEW: Graph class + 3 graph algorithms
│   ├── sorting.py            # Existing: 6 sorting algorithms
│   └── base.py               # Existing: Algorithm protocol
└── data/
    ├── graph_generators.py   # NEW: 5 graph generator functions
    ├── generators.py         # Existing: Array data generators
    └── __init__.py           # Updated: Exports new generators
```

## 1. Searching Algorithms (`complexity-profiler/algorithms/searching.py`)

### Linear Search - O(n)

Simple sequential search that examines each element until target is found.

```python
from complexity-profiler.algorithms.searching import LinearSearch
from complexity-profiler.analysis.metrics import DefaultMetricsCollector

# Create algorithm instance
linear_search = LinearSearch[int]()

# Prepare data: [target, ...search_space]
data = [5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Execute search
collector = DefaultMetricsCollector()
result = linear_search.execute(data, collector)
print(f"Found at index: {result[0]}")  # Output: 4

# Get metrics
metrics = collector.get_metrics()
print(f"Comparisons: {metrics.comparisons}")
print(f"Array accesses: {metrics.accesses}")
```

**Algorithm Properties:**
- Time Complexity: O(n) average and worst case, O(1) best case
- Space Complexity: O(1)
- Works on unsorted data
- No preprocessing required

### Binary Search - O(log n)

Efficient divide-and-conquer search on sorted data.

```python
from complexity-profiler.algorithms.searching import BinarySearch

binary_search = BinarySearch[int]()

# Data must be sorted: [target, ...sorted_array]
data = [7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

collector = DefaultMetricsCollector()
result = binary_search.execute(data, collector)
print(f"Found at index: {result[0]}")  # Output: 6
```

**Algorithm Properties:**
- Time Complexity: O(log n)
- Space Complexity: O(1) (iterative)
- Requires sorted data
- Much faster than linear search for large datasets

### Jump Search - O(√n)

Block-based search that jumps ahead by √n steps.

```python
from complexity-profiler.algorithms.searching import JumpSearch

jump_search = JumpSearch[int]()

# Data must be sorted
data = [8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

collector = DefaultMetricsCollector()
result = jump_search.execute(data, collector)
print(f"Found at index: {result[0]}")  # Output: 7
```

**Algorithm Properties:**
- Time Complexity: O(√n)
- Space Complexity: O(1)
- Requires sorted data
- Better than linear, worse than binary
- Good when jumping backward is expensive

### Interpolation Search - O(log log n) average

Position-based search using interpolation formula.

```python
from complexity-profiler.algorithms.searching import InterpolationSearch

interp_search = InterpolationSearch[int]()

# Works best on uniformly distributed sorted data
data = [5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

collector = DefaultMetricsCollector()
result = interp_search.execute(data, collector)
print(f"Found at index: {result[0]}")  # Output: 4
```

**Algorithm Properties:**
- Time Complexity: O(log log n) average, O(n) worst case
- Space Complexity: O(1)
- Requires sorted, uniformly distributed data
- Can be faster than binary search for uniform data

## 2. Graph Algorithms (`complexity-profiler/algorithms/graph.py`)

### Graph Data Structure

The `Graph` class uses adjacency list representation and supports both directed and undirected graphs.

```python
from complexity-profiler.algorithms.graph import Graph

# Create an undirected graph
graph = Graph[int](directed=False)

# Add edges (automatically adds vertices)
graph.add_edge(0, 1, weight=2.0)
graph.add_edge(0, 2, weight=3.0)
graph.add_edge(1, 3, weight=1.0)
graph.add_edge(2, 3, weight=4.0)
graph.add_edge(3, 4, weight=2.5)

print(f"Vertices: {graph.get_vertex_count()}")  # Output: 5
print(f"Edges: {graph.get_edge_count()}")      # Output: 5

# Get neighbors
neighbors = graph.get_neighbors(1)
print(f"Neighbors of 1: {neighbors}")  # [(0, 2.0), (3, 1.0)]
```

### Breadth-First Search (BFS) - O(V + E)

Level-by-level graph traversal using a queue.

```python
from complexity-profiler.algorithms.graph import BFS
from complexity-profiler.analysis.metrics import DefaultMetricsCollector

# Create graph
graph = Graph[int]()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 3)
graph.add_edge(2, 3)
graph.add_edge(3, 4)

# Execute BFS
bfs = BFS[int]()
collector = DefaultMetricsCollector()
traversal = bfs.execute([graph], collector)

print(f"BFS Traversal: {traversal}")  # Example: [0, 1, 2, 3, 4]
print(f"Operations: {collector.get_metrics().total_operations()}")
```

**Algorithm Properties:**
- Time Complexity: O(V + E) where V = vertices, E = edges
- Space Complexity: O(V)
- Finds shortest path in unweighted graphs
- Explores level by level

**Use Cases:**
- Shortest path in unweighted graphs
- Web crawling
- Social network analysis
- Finding connected components

### Depth-First Search (DFS) - O(V + E)

Deep exploration using recursion or stack.

```python
from complexity-profiler.algorithms.graph import DFS

# Create graph
graph = Graph[int]()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 3)
graph.add_edge(2, 3)
graph.add_edge(3, 4)

# Execute DFS
dfs = DFS[int]()
collector = DefaultMetricsCollector()
traversal = dfs.execute([graph], collector)

print(f"DFS Traversal: {traversal}")  # Example: [0, 1, 3, 2, 4]
```

**Algorithm Properties:**
- Time Complexity: O(V + E)
- Space Complexity: O(V)
- Explores deeply before backtracking
- Uses recursion

**Use Cases:**
- Topological sorting
- Cycle detection
- Maze solving
- Pathfinding

### Dijkstra's Shortest Path - O((V + E) log V)

Single-source shortest path using priority queue.

```python
from complexity-profiler.algorithms.graph import DijkstraShortestPath

# Create weighted graph
graph = Graph[int]()
graph.add_edge(0, 1, weight=4.0)
graph.add_edge(0, 2, weight=1.0)
graph.add_edge(2, 1, weight=2.0)
graph.add_edge(1, 3, weight=1.0)
graph.add_edge(2, 3, weight=5.0)
graph.add_edge(3, 4, weight=3.0)

# Execute Dijkstra
dijkstra = DijkstraShortestPath[int]()
collector = DefaultMetricsCollector()
distances = dijkstra.execute([graph], collector)

print("Shortest paths from source:")
for vertex, distance in distances:
    print(f"  To {vertex}: {distance}")
```

**Algorithm Properties:**
- Time Complexity: O((V + E) log V) with binary heap
- Space Complexity: O(V)
- Requires non-negative edge weights
- Finds shortest paths to all reachable vertices

**Use Cases:**
- GPS navigation
- Network routing
- Game AI pathfinding
- Resource allocation

## 3. Graph Generators (`complexity-profiler/data/graph_generators.py`)

### Random Graph

Generate random graphs with specified edge probability.

```python
from complexity-profiler.data.graph_generators import random_graph

# Create random graph with 10 vertices, 30% edge probability
graph = random_graph(
    vertices=10,
    edge_probability=0.3,
    directed=False,
    weighted=True,
    min_weight=1.0,
    max_weight=10.0,
    seed=42  # For reproducibility
)

print(graph)  # Graph(vertices=10, edges=~13, directed=False)
```

**Parameters:**
- `vertices`: Number of vertices
- `edge_probability`: Probability (0.0-1.0) of edge existing
- `directed`: Whether graph is directed
- `weighted`: Whether to assign random weights
- `min_weight`, `max_weight`: Weight range if weighted
- `seed`: Random seed for reproducibility

### Complete Graph

Every vertex connected to every other vertex.

```python
from complexity-profiler.data.graph_generators import complete_graph

# Create complete graph with 5 vertices
graph = complete_graph(
    vertices=5,
    directed=False
)

# Undirected complete graph with n vertices has n(n-1)/2 edges
print(graph.get_edge_count())  # Output: 10
```

**Formula:**
- Undirected: n(n-1)/2 edges
- Directed: n(n-1) edges

### Tree Graph

Connected acyclic graph with n-1 edges.

```python
from complexity-profiler.data.graph_generators import tree_graph

# Create tree with 10 vertices
graph = tree_graph(
    vertices=10,
    branching_factor=3,  # Max children per node
    weighted=False,
    seed=42
)

print(graph.get_edge_count())  # Output: 9 (always n-1 for trees)
```

**Properties:**
- Always n-1 edges for n vertices
- Connected and acyclic
- Undirected

### Linear Graph (Path Graph)

Vertices form a single straight path.

```python
from complexity-profiler.data.graph_generators import linear_graph

# Create linear graph: 0 -- 1 -- 2 -- 3 -- 4
graph = linear_graph(vertices=5)

print(graph.get_edge_count())  # Output: 4
```

### Cyclic Graph

Vertices form a closed cycle.

```python
from complexity-profiler.data.graph_generators import cyclic_graph

# Create cycle: 0 -- 1 -- 2 -- 3 -- 4 -- 0
graph = cyclic_graph(vertices=5)

print(graph.get_edge_count())  # Output: 5 (n edges for n vertices)
```

## Integration with Existing System

All Phase 4 components integrate seamlessly with the existing analyzer:

```python
from complexity-profiler.algorithms import BinarySearch, BFS
from complexity-profiler.data import random_graph
from complexity-profiler.analysis.metrics import DefaultMetricsCollector

# Example 1: Analyze Binary Search
search_algo = BinarySearch[int]()
print(f"Algorithm: {search_algo.metadata.name}")
print(f"Category: {search_algo.metadata.category}")
print(f"Complexity: {search_algo.metadata.expected_complexity}")

# Example 2: Analyze BFS on generated graph
graph = random_graph(vertices=100, edge_probability=0.1, seed=42)
bfs_algo = BFS[int]()
collector = DefaultMetricsCollector()
collector.start_timing()
result = bfs_algo.execute([graph], collector)
collector.stop_timing()

metrics = collector.get_metrics()
print(f"Vertices visited: {len(result)}")
print(f"Total operations: {metrics.total_operations()}")
print(f"Execution time: {metrics.execution_time:.6f}s")
```

## Testing

A comprehensive test suite is provided in `test_phase4.py`:

```bash
python test_phase4.py
```

This validates:
- All 4 searching algorithms work correctly
- All 3 graph algorithms traverse graphs properly
- All 5 graph generators create valid graphs
- Metrics collection works for all algorithms
- Algorithm metadata is properly defined

## Summary

Phase 4 successfully expands the Big-O Complexity Analyzer with:

### Searching Algorithms (4)
1. **LinearSearch** - O(n) - Sequential search
2. **BinarySearch** - O(log n) - Divide and conquer
3. **JumpSearch** - O(√n) - Block jumping
4. **InterpolationSearch** - O(log log n) avg - Position estimation

### Graph Algorithms (3 + Graph structure)
1. **Graph** - Adjacency list representation
2. **BFS** - O(V + E) - Breadth-first traversal
3. **DFS** - O(V + E) - Depth-first traversal
4. **DijkstraShortestPath** - O((V + E) log V) - Shortest paths

### Graph Generators (5)
1. **random_graph** - Random edges based on probability
2. **complete_graph** - Fully connected graph
3. **tree_graph** - Connected acyclic graph
4. **linear_graph** - Single path
5. **cyclic_graph** - Closed cycle

All implementations:
- Follow the `Algorithm` protocol from `base.py`
- Include comprehensive type hints
- Have detailed docstrings with complexity analysis
- Collect metrics via `MetricsCollector`
- Are production-quality with proper error handling
- Include usage examples and edge case handling

The implementation is complete and ready for integration with the rest of the Big-O Complexity Analyzer system.
