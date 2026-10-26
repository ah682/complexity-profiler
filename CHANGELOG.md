# Release notes

## 1.0.1 — 26 October 2026

- Correct timing-scale-dependent R² and enforce the minimum acceptable fit score.
- Use normalized affine least squares with nonnegative growth coefficients and
  explicit invalid-input checks. Fix the cubic enum alias.
- Report ambiguous/poor fits explicitly, preserving raw measurements and candidate
  scores. Direct fitting can now raise its documented RuntimeError for inconclusive
  evidence. JSON round trips restore nested metrics and statistics correctly.
- Repair package imports, CLI comparison fields, type annotations, formatting and
  narrowly document non-security uses of pseudorandom data for Bandit.
- Replace exact wall-clock complexity assertions with deterministic synthetic
  fitting regressions and keep real timing/operation-count smoke coverage.
- Add an installable root README, real repository metadata and contributor checks.

- Stabilize polynomial bases for extreme finite sizes, normalize nonzero basis
  magnitudes without an absolute floor, and report unrepresentable candidates as
  inconclusive. Public curve-function coefficients remain unchanged.

- Collect Phase 1 checks as pytest tests and verify timing against exact clock
  differences rather than platform-specific sleep durations. Preserve real-clock
  smoke coverage and the full 85% coverage gate.
- Keep package, module and CLI version reporting in agreement for the release.
