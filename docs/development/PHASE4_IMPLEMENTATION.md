# Search and graph implementation notes

The algorithm expansion added four searches (`LinearSearch`, `BinarySearch`,
`JumpSearch`, `InterpolationSearch`), three graph algorithms (`BFS`, `DFS`,
`DijkstraShortestPath`) and an adjacency-list `Graph` class.

Searches accept `[target, *search_space]` and return `[index]` or `[-1]`.
Binary, jump and interpolation search expect sorted data. They are registered
with the CLI alongside the six sorting algorithms.

Graph algorithms accept `[graph]` through the Python API. Their public execution
methods select a vertex from the graph's vertex set. The generators are
`random_graph`, `complete_graph`, `tree_graph`, `linear_graph` and `cyclic_graph`.
For five vertices, an undirected complete graph has 10 edges, a tree or path has
four, and a cycle has five.

See the [algorithm reference](../algorithms.md) for examples and interface details.
Tests are in [test_phase4.py](../../tests/test_phase4.py); run them with
`pytest tests/test_phase4.py --no-cov` during development, then run `pytest` for
the full coverage check.
