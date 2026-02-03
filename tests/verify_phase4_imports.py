"""
Quick import verification for Phase 4 implementations.
"""

print("Verifying Phase 4 imports...")
print("-" * 60)

try:
    # Test searching algorithms imports
    from complexity_profiler.algorithms.searching import (
        LinearSearch,
        BinarySearch,
        JumpSearch,
        InterpolationSearch,
    )
    print("✓ Searching algorithms imported successfully")
    print(f"  - LinearSearch: {LinearSearch}")
    print(f"  - BinarySearch: {BinarySearch}")
    print(f"  - JumpSearch: {JumpSearch}")
    print(f"  - InterpolationSearch: {InterpolationSearch}")
except ImportError as e:
    print(f"✗ Failed to import searching algorithms: {e}")

print()

try:
    # Test graph algorithms imports
    from complexity_profiler.algorithms.graph import (
        Graph,
        BFS,
        DFS,
        DijkstraShortestPath,
    )
    print("✓ Graph algorithms imported successfully")
    print(f"  - Graph: {Graph}")
    print(f"  - BFS: {BFS}")
    print(f"  - DFS: {DFS}")
    print(f"  - DijkstraShortestPath: {DijkstraShortestPath}")
except ImportError as e:
    print(f"✗ Failed to import graph algorithms: {e}")

print()

try:
    # Test graph generators imports
    from complexity_profiler.data.graph_generators import (
        random_graph,
        complete_graph,
        tree_graph,
        linear_graph,
        cyclic_graph,
    )
    print("✓ Graph generators imported successfully")
    print(f"  - random_graph: {random_graph}")
    print(f"  - complete_graph: {complete_graph}")
    print(f"  - tree_graph: {tree_graph}")
    print(f"  - linear_graph: {linear_graph}")
    print(f"  - cyclic_graph: {cyclic_graph}")
except ImportError as e:
    print(f"✗ Failed to import graph generators: {e}")

print()

try:
    # Test package-level imports
    from complexity_profiler.algorithms import (
        LinearSearch as LS,
        BinarySearch as BS,
        BFS as BFSAlgo,
        Graph as G,
    )
    from complexity_profiler.data import (
        random_graph as rg,
        complete_graph as cg,
    )
    print("✓ Package-level imports work correctly")
    print(f"  - complexity_profiler.algorithms exports search/graph algorithms")
    print(f"  - complexity_profiler.data exports graph generators")
except ImportError as e:
    print(f"✗ Failed package-level imports: {e}")

print()
print("-" * 60)
print("Import verification complete!")
