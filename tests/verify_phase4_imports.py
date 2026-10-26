"""
Quick import verification for Phase 4 implementations.
"""

print("Verifying Phase 4 imports...")
print("-" * 60)

try:
    # Test searching algorithms imports
    from complexity_profiler.algorithms.searching import (
        BinarySearch,
        InterpolationSearch,
        JumpSearch,
        LinearSearch,
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
        BFS,
        DFS,
        DijkstraShortestPath,
        Graph,
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
        complete_graph,
        cyclic_graph,
        linear_graph,
        random_graph,
        tree_graph,
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
    from complexity_profiler.algorithms import BFS as BFSAlgo
    from complexity_profiler.algorithms import BinarySearch as BS
    from complexity_profiler.algorithms import Graph as G
    from complexity_profiler.algorithms import LinearSearch as LS
    from complexity_profiler.data import complete_graph as cg
    from complexity_profiler.data import random_graph as rg

    for exported in (LS, BS, BFSAlgo, G, rg, cg):
        assert callable(exported)
    print("✓ Package-level imports work correctly")
    print("  - complexity_profiler.algorithms exports search/graph algorithms")
    print("  - complexity_profiler.data exports graph generators")
except ImportError as e:
    print(f"✗ Failed package-level imports: {e}")

print()
print("-" * 60)
print("Import verification complete!")
