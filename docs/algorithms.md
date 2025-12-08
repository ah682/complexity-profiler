# Algorithm Reference

Reference for all implemented algorithms.

## Sorting Algorithms

### Merge Sort

**Complexity**: O(n log n) - all cases
**Space**: O(n)
**Stable**: Yes
**In-place**: No

Divide-and-conquer algorithm that recursively splits the array, sorts the halves, and merges them. Guarantees O(n log n) regardless of input.

**When to use**:
- Need guaranteed O(n log n) performance
- Stability is required
- Working with linked lists
- External sorting (large datasets)

**Usage**:
```bash
complexity-profiler analyze merge_sort --sizes 100,1000,10000
```

---

### Quick Sort

**Complexity**: O(n log n) average, O(n�) worst
**Space**: O(log n) to O(n)
**Stable**: No
**In-place**: No (this implementation)

Picks a pivot and partitions the array into elements less than, equal to, and greater than the pivot. Uses random pivot selection to avoid worst-case behavior on sorted data.

**When to use**:
- Average-case performance is prioritized
- In-memory sorting with good cache locality
- Don't need stability

**Usage**:
```bash
complexity-profiler analyze quick_sort --data-type random
```

---

### Heap Sort

**Complexity**: O(n log n) - all cases
**Space**: O(1)
**Stable**: No
**In-place**: Yes

Builds a max heap and repeatedly extracts the maximum element. Gets you guaranteed O(n log n) with O(1) space.

**When to use**:
- Need O(n log n) with minimal memory
- Guaranteed performance important
- Stability not required

**Usage**:
```bash
complexity-profiler analyze heap_sort --sizes 1000,5000,10000
```

---

### Insertion Sort

**Complexity**: O(n�) worst/average, O(n) best
**Space**: O(1)
**Stable**: Yes
**In-place**: Yes

Builds the sorted array one element at a time by inserting each element into its correct position. Works well on small or nearly-sorted arrays.

**When to use**:
- Small datasets (< 50 elements)
- Nearly sorted data
- Online sorting (processing stream)
- As part of hybrid algorithms (TimSort)

**Usage**:
```bash
complexity-profiler analyze insertion_sort --data-type nearly_sorted
```

---

### Bubble Sort

**Complexity**: O(n�) worst/average, O(n) best
**Space**: O(1)
**Stable**: Yes
**In-place**: Yes

Steps through the list, compares adjacent elements, and swaps them if needed. Includes early termination when no swaps occur.

**When to use**:
- Educational purposes
- Very small datasets
- Need simple, stable sorting
- Already mostly sorted data

**Usage**:
```bash
complexity-profiler analyze bubble_sort --sizes 100,500,1000
```

---

### Selection Sort

**Complexity**: O(n�) - all cases
**Space**: O(1)
**Stable**: No
**In-place**: Yes

Finds the minimum element from the unsorted portion and puts it at the beginning. Minimizes the number of swaps.

**When to use**:
- Minimize number of swaps (expensive writes)
- Simple implementation needed
- Memory is extremely limited

**Usage**:
```bash
complexity-profiler analyze selection_sort
```

---

## Searching Algorithms

### Binary Search

**Complexity**: O(log n)
**Space**: O(1)
**Requirements**: Sorted array

Divides the search space in half by comparing the target with the middle element. Requires sorted data.

**When to use**:
- Searching in sorted data
- Large datasets
- Need logarithmic performance

**Usage**:
```bash
complexity-profiler analyze binary_search --data-type sorted
```

---

### Linear Search

**Complexity**: O(n)
**Space**: O(1)
**Requirements**: None

**Description**: Sequentially checks each element until target is found or end is reached.

**When to use**:
- Unsorted data
- Small datasets
- Simple implementation needed
- Single search operation

**Usage**:
```bash
complexity-profiler analyze linear_search
```

---

### Jump Search

**Complexity**: O(n)
**Space**: O(1)
**Requirements**: Sorted array

Jumps ahead by fixed steps, then does linear search in the identified block. Faster than linear search, simpler than binary.

**When to use**:
- Sorted data
- System where jumping back is costly
- Simpler than binary search

**Usage**:
```bash
complexity-profiler analyze jump_search --data-type sorted
```

---

### Interpolation Search

**Complexity**: O(log log n) average, O(n) worst
**Space**: O(1)
**Requirements**: Sorted array with uniform distribution

Better than binary search for uniformly distributed data. Uses value-based position estimation instead of always picking the middle.

**When to use**:
- Uniformly distributed sorted data
- Need faster than O(log n)
- Data allows value interpolation

**Usage**:
```bash
complexity-profiler analyze interpolation_search --data-type sorted
```

---

## Graph Algorithms

### Breadth-First Search (BFS)

**Complexity**: O(V + E) where V = vertices, E = edges
**Space**: O(V)

Explores the graph level by level using a queue. Finds the shortest path in unweighted graphs.

**When to use**:
- Shortest path in unweighted graph
- Level-order traversal
- Finding connected components

---

### Depth-First Search (DFS)

**Complexity**: O(V + E)
**Space**: O(V)

Goes as deep as possible before backtracking. Can use recursion or an explicit stack.

**When to use**:
- Topological sorting
- Cycle detection
- Path finding
- Maze solving

---

### Dijkstra's Shortest Path

**Complexity**: O((V + E) log V) with binary heap
**Space**: O(V)

Finds shortest paths from a source to all vertices in a weighted graph. Requires non-negative edge weights.

**When to use**:
- Weighted graphs
- Single-source shortest path
- Non-negative edge weights
- GPS navigation, network routing

---

## Comparison Tables

### Sorting Algorithms Comparison

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable | In-place |
|-----------|-------------|------------|--------------|-------|--------|----------|
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
| Quick Sort | O(n log n) | O(n log n) | O(n�) | O(log n) | No | No* |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes |
| Insertion Sort | O(n) | O(n�) | O(n�) | O(1) | Yes | Yes |
| Bubble Sort | O(n) | O(n�) | O(n�) | O(1) | Yes | Yes |
| Selection Sort | O(n�) | O(n�) | O(n�) | O(1) | No | Yes |

### Searching Algorithms Comparison

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Requirements |
|-----------|-------------|------------|--------------|-------|--------------|
| Binary Search | O(1) | O(log n) | O(log n) | O(1) | Sorted |
| Linear Search | O(1) | O(n) | O(n) | O(1) | None |
| Jump Search | O(1) | O(n) | O(n) | O(1) | Sorted |
| Interpolation | O(1) | O(log log n) | O(n) | O(1) | Sorted, uniform |

---

## Algorithm Selection Guide

### For Sorting:

- Need guaranteed O(n log n)? Use Merge Sort or Heap Sort
- Limited memory? Use Heap Sort
- Need stability? Use Merge Sort, Insertion Sort, or Bubble Sort
- Small dataset (< 50 elements)? Use Insertion Sort
- General purpose? Use Quick Sort (average case) or Merge Sort
- Nearly sorted data? Use Insertion Sort or Bubble Sort

### For Searching:

- Sorted data? Use Binary Search
- Unsorted data? Use Linear Search
- Sorted + uniform distribution? Use Interpolation Search
- Sorted + simpler than binary? Use Jump Search

---

## Running Comparisons

Compare multiple algorithms:
```bash
# Compare sorting algorithms
complexity-profiler compare merge_sort quick_sort heap_sort \
  --sizes 100,1000,10000 \
  --runs 20

# Compare searching algorithms
complexity-profiler compare binary_search linear_search jump_search \
  --data-type sorted
```

Export results:
```bash
complexity-profiler analyze merge_sort \
  --export json \
  --output merge_sort_results.json
```

Visualize:
```bash
complexity-profiler analyze quick_sort \
  --save-chart quick_sort_analysis.png
```

---

For implementation details, see the source code in `complexity-profiler/algorithms/`.
