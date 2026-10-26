# Complexity Profiler documentation

Start with the [project README](../README.md) to install the package and run a
profile. These guides cover the CLI, Python API and implementation:

- [Basic usage](examples/basic_usage.md): run, compare, chart and export profiles.
- [Custom experiments](examples/advanced_analysis.md): control inputs and inspect
  operation counts and candidate scores.
- [Algorithms](algorithms.md): sorting, searching and graph interfaces.
- [Export guide](EXPORT_GUIDE.md): JSON round trips, CSV tables and error handling.
- [Architecture](ARCHITECTURE.md): how profiling, fitting and presentation fit together.
- [Contributing](CONTRIBUTING.md) and [tests](../tests/README.md): development setup
  and regression checks.

The package is imported as `complexity_profiler`; the installed command is
`complexity-profiler`. Run `complexity-profiler --help` or add `--help` to a
subcommand to see its options.

The [development notes](development/IMPLEMENTATION_SUMMARY.md) describe the original
implementation stages. Release changes are in the [changelog](../CHANGELOG.md).
