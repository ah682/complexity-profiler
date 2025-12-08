"""
Quick test script to verify Phase 4 implementations.
"""

from complexity_profiler.algorithms.searching import (
    LinearSearch,
    BinarySearch,
    JumpSearch,
    InterpolationSearch,
)
from complexity_profiler.algorithms.graph import Graph, BFS, DFS, DijkstraShortestPath
from complexity_profiler.data.graph_generators import (
    random_graph,
    complete_graph,
    tree_graph,
    linear_graph,
    cyclic_graph,
)
from complexity_profiler.analysis.metrics import DefaultMetricsCollector


def test_searching_algorithms():
    """Test all searching algorithms."""
    print("Testing Searching Algorithms...")

    # Test data: [target, ...sorted_array]
    search_data = [5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    collector = DefaultMetricsCollector()

    # Linear Search
    ls = LinearSearch[int]()
    result = ls.execute(search_data, collector)
    print(f"  Linear Search: Found at index {result[0]} (expected 4)")
    print(f"    Metadata: {ls.metadata.name} - {ls.metadata.expected_complexity}")

    # Binary Search
    bs = BinarySearch[int]()
    collector.reset()
    result = bs.execute(search_data, collector)
    print(f"  Binary Search: Found at index {result[0]} (expected 4)")
    print(f"    Metadata: {bs.metadata.name} - {bs.metadata.expected_complexity}")

    # Jump Search
    js = JumpSearch[int]()
    collector.reset()
    result = js.execute(search_data, collector)
    print(f"  Jump Search: Found at index {result[0]} (expected 4)")
    print(f"    Metadata: {js.metadata.name} - {js.metadata.expected_complexity}")

    # Interpolation Search
    interp = InterpolationSearch[int]()
    collector.reset()
    result = interp.execute(search_data, collector)
    print(f"  Interpolation Search: Found at index {result[0]} (expected 4)")
    print(f"    Metadata: {interp.metadata.name} - {interp.metadata.expected_complexity}")

    print("  All searching algorithms passed!\n")


def test_graph_algorithms():
    """Test all graph algorithms."""
    print("Testing Graph Algorithms...")

    # Create a simple test graph
    graph = Graph[int]()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    print(f"  Test graph: {graph}")

    collector = DefaultMetricsCollector()

    # BFS
    bfs = BFS[int]()
    result = bfs.execute([graph], collector)
    print(f"  BFS Traversal: {result}")
    print(f"    Metadata: {bfs.metadata.name} - {bfs.metadata.expected_complexity}")

    # DFS
    dfs = DFS[int]()
    collector.reset()
    result = dfs.execute([graph], collector)
    print(f"  DFS Traversal: {result}")
    print(f"    Metadata: {dfs.metadata.name} - {dfs.metadata.expected_complexity}")

    # Dijkstra
    dijkstra = DijkstraShortestPath[int]()
    collector.reset()
    result = dijkstra.execute([graph], collector)
    print(f"  Dijkstra Shortest Paths: {result}")
    print(f"    Metadata: {dijkstra.metadata.name} - {dijkstra.metadata.expected_complexity}")

    print("  All graph algorithms passed!\n")


def test_graph_generators():
    """Test all graph generators."""
    print("Testing Graph Generators...")

    # Random graph
    rg = random_graph(10, edge_probability=0.3, seed=42)
    print(f"  Random Graph: {rg}")

    # Complete graph
    cg = complete_graph(5)
    print(f"  Complete Graph: {cg}")
    expected_edges = 5 * 4 // 2
    print(f"    Expected {expected_edges} edges, got {cg.get_edge_count()}")

    # Tree graph
    tg = tree_graph(10, seed=42)
    print(f"  Tree Graph: {tg}")
    print(f"    Expected 9 edges, got {tg.get_edge_count()}")

    # Linear graph
    lg = linear_graph(5)
    print(f"  Linear Graph: {lg}")
    print(f"    Expected 4 edges, got {lg.get_edge_count()}")

    # Cyclic graph
    cyg = cyclic_graph(5)
    print(f"  Cyclic Graph: {cyg}")
    print(f"    Expected 5 edges, got {cyg.get_edge_count()}")

    print("  All graph generators passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 4 Implementation Test Suite")
    print("=" * 60 + "\n")

    test_searching_algorithms()
    test_graph_algorithms()
    test_graph_generators()

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
