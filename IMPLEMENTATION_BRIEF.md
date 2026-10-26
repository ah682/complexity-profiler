# Verified profiler repair brief

Baseline: 055f314fe2b5278f1a745889748669d86a8b423a, branch quality/verified-profiler.

Reproduce the configured CI checks first, then repair package imports, numerical fitting,
and genuine lint/type/security errors. Preserve the successful tuple API; use its documented
RuntimeError contract for an inconclusive fit, carrying candidate scores in a dedicated
subclass. Retain raw profiling measurements when fit evidence is insufficient. Fit only
nonnegative growth coefficients, normalize numerical scales, validate data and thresholds,
and make ambiguity visible instead of silently naming a near-tied candidate.
Ambiguity uses both a 0.02 R² gap and a residual ratio of at most two, with a
64-epsilon floor, to distinguish exact synthetic curves from noisy near ties.

Regressions will use hand-derived measurements across timing units, pathological inputs,
weak fits and ambiguous ranges. Integration correctness will use deterministic measurements;
real timing checks will verify measurement collection rather than exact theoretical classes.
Preserve MIT. Add an accurate root README, release notes and installable metadata. Run all
configured checks and install a built wheel into a separate environment outside the checkout.
