# Implementation notes

Complexity Profiler grew from a timing script into a package with interchangeable
algorithms, operation collectors, statistical summaries, growth fitting, a CLI,
charts and exports. The original work was organized into these stages:

1. Algorithm and collector protocols, metadata and input generators.
2. Repeated profiling, statistics and candidate curve fitting.
3. Click commands, Rich tables and Matplotlib charts.
4. Four search algorithms, three graph algorithms and five graph generators.
5. JSON/CSV exports, packaging and contributor documentation.

The CLI exposes six sorting and four searching algorithms. Graph algorithms use
the Python API. The current test configuration enforces 85% coverage; coverage
and runtime figures should come from the run being reviewed.

The 1.0.1 update added timing-unit-independent fits, invalid-input checks and
explicit inconclusive results. It also repaired nested JSON loading, package
imports, CLI comparisons and distribution checks. Details are in the
[release notes](../../CHANGELOG.md).

For current use, see the [architecture](../ARCHITECTURE.md),
[algorithm reference](../algorithms.md), [export guide](../EXPORT_GUIDE.md) and
[test guide](../../tests/README.md). Earlier implementation reports remain in Git
history; their line counts and completion checklists are no longer maintained.
