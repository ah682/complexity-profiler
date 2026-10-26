# Numerical fitting repair notes

The 1.0.1 work started from `055f314fe2b5278f1a745889748669d86a8b423a`.
It repaired package imports, curve fitting, exports and the configured development
checks while preserving the successful `(ComplexityClass, r_squared)` fitter API.

The key design choice was to keep raw measurements when a fit is poor or
ambiguous. Direct fitting uses `InconclusiveFitError`, a `RuntimeError` subclass
with candidate scores; profiling stores that outcome in `ProfileResult`.

Fits use nonnegative growth coefficients and normalized numerical scales.
Ambiguity requires both an R² gap of at most 0.02 and a residual ratio of at
most two, with a 64-machine-epsilon floor. Regressions use hand-derived curves,
invalid inputs and multiple timing units. Real-clock checks cover collection
separately from fitting correctness.

See the [architecture](docs/ARCHITECTURE.md#fitting),
[test guide](tests/README.md) and [release notes](CHANGELOG.md) for the current
implementation and checks. The project retains its MIT licence.
