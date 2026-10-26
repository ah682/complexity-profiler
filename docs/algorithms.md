# Algorithms

The CLI includes six sorting and four searching algorithms. BFS, DFS and
Dijkstra's algorithm are available through the Python API. Each implementation
records operations through a supplied metrics collector.

## Sorting

| CLI name | Approach | Usual time bound |
| --- | --- | --- |
| `merge_sort` | Split, recursively sort and merge. | O(n log n) |
| `quick_sort` | Random pivot with less/equal/greater partitions. | O(n log n) average; O(n²) worst |
| `heap_sort` | Build a max heap and remove its largest element repeatedly. | O(n log n) |
| `insertion_sort` | Insert each value into the sorted prefix. | O(n²) average/worst; O(n) on sorted input |
| `bubble_sort` | Swap adjacent values, stopping when a pass makes no swaps. | O(n²) average/worst; O(n) on sorted input |
| `selection_sort` | Select the next minimum from the unsorted suffix. | O(n²) |

These are theoretical reference bounds. Measured timings also include Python
and instrumentation overhead. All public sorting `execute` methods copy their
input. Heap, insertion, bubble and selection sort then modify that working copy;
merge and quick sort allocate additional lists. An algorithm's in-place metadata
therefore describes its working procedure, not a promise to mutate the caller's
list or use constant total memory through this API.

```bash
complexity-profiler compare merge_sort quick_sort heap_sort --sizes 100,500,1000,5000 --runs 10 --save-chart sorting.png
```

## Searching

| CLI name | Input requirement | Usual time bound |
| --- | --- | --- |
| `linear_search` | Any list. | O(n) worst |
| `binary_search` | Sorted list. | O(log n) worst |
| `jump_search` | Sorted list. | O(√n) worst |
| `interpolation_search` | Sorted numeric list; uniform spacing helps. | O(log log n) average for uniform data; O(n) worst |

The input format is `[target, *search_space]`. The result is a one-element list
containing the matching index in `search_space`, or `-1`. See
[custom experiments](examples/advanced_analysis.md#search-inputs) for a small
Python example and guidance on target selection. The CLI's generic data generator
does not define an average-case search experiment.

## Graphs

`Graph` stores an adjacency list and supports directed and weighted edges.
`BFS`, `DFS` and `DijkstraShortestPath` accept a list containing one `Graph`.
Their public `execute` methods choose a starting vertex from the graph's vertex
set. BFS and DFS return traversal order; Dijkstra returns vertex/distance pairs
and requires nonnegative weights. A set's iteration order should not be used to
choose a particular source for an experiment.

```python
from complexity_profiler.algorithms.graph import BFS
from complexity_profiler.analysis.metrics import DefaultMetricsCollector
from complexity_profiler.data.graph_generators import linear_graph

graph = linear_graph(vertices=5)
collector = DefaultMetricsCollector()
visited = BFS().execute([graph], collector)
print(visited)
print(graph.get_vertex_count(), graph.get_edge_count())  # 5, 4
```

Generators in `complexity_profiler.data.graph_generators` include `random_graph`,
`complete_graph`, `tree_graph`, `linear_graph` and `cyclic_graph`. They accept a
vertex count; random and tree generation also support a seed. A five-vertex
undirected complete graph has 10 edges, a tree or path has four, and a cycle
has five. See their function signatures for direction and weight options.

The graph implementations and tests are in
[`graph.py`](../complexity_profiler/algorithms/graph.py) and
[`test_phase4.py`](../tests/test_phase4.py).
