# Export implementation history

The original export work added JSON and CSV writers, JSON profile restoration,
CSV loading, a summary export and package-level imports. It also added
`examples/export_example.py` and `examples/test_export_functionality.py`.

Current usage and behavior are documented in the [export guide](../EXPORT_GUIDE.md).
The maintained regression suite includes
[output integration tests](../../tests/integration/test_outputs.py) and
[fit/export cases](../../tests/integration/test_fit_evidence.py).

This file keeps the original implementation-note URL available. The detailed
completion checklist is retained in Git history; current checks are described
in the [test guide](../../tests/README.md).
