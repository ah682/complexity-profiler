"""
Quick verification script for Phase 1 implementation.
Tests the three core modules: exceptions, logging_config, and metrics.
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Phase 1 Implementation Verification")
print("=" * 70)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from complexity_profiler.utils.exceptions import (
        BigOComplexityError,
        AlgorithmError,
        DataGenerationError,
        ProfilingError,
        CurveFittingError,
        ConfigurationError,
        ExportError,
    )
    print("   ✓ exceptions module imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import exceptions: {e}")
    sys.exit(1)

try:
    from complexity_profiler.utils.logging_config import (
        setup_logging,
        get_logger,
        set_module_level,
    )
    print("   ✓ logging_config module imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import logging_config: {e}")
    sys.exit(1)

try:
    from complexity_profiler.analysis.metrics import (
        PerformanceMetrics,
        DefaultMetricsCollector,
        ThreadSafeMetricsCollector,
        AggregateMetricsCollector,
        measure_time,
    )
    print("   ✓ metrics module imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import metrics: {e}")
    sys.exit(1)

# Test 2: Test exception hierarchy
print("\n2. Testing exception hierarchy...")
try:
    # Test base exception
    try:
        raise BigOComplexityError("Test error", details={"key": "value"})
    except BigOComplexityError as e:
        assert str(e) == "Test error (key=value)"

    # Test AlgorithmError
    try:
        raise AlgorithmError("Algorithm failed", algorithm_name="quicksort")
    except AlgorithmError as e:
        assert e.algorithm_name == "quicksort"

    print("   ✓ Exception hierarchy working correctly")
except Exception as e:
    print(f"   ✗ Exception tests failed: {e}")
    sys.exit(1)

# Test 3: Test logging configuration
print("\n3. Testing logging configuration...")
try:
    logger = setup_logging(level="INFO", enable_rich=False)
    assert logger is not None

    module_logger = get_logger(__name__)
    assert module_logger is not None

    set_module_level("test_module", "DEBUG")

    print("   ✓ Logging configuration working correctly")
except Exception as e:
    print(f"   ✗ Logging tests failed: {e}")
    sys.exit(1)

# Test 4: Test metrics collection
print("\n4. Testing metrics collection...")
try:
    # Test PerformanceMetrics
    metrics = PerformanceMetrics(comparisons=10, swaps=5)
    assert metrics.comparisons == 10
    assert metrics.total_operations() == 15

    # Test DefaultMetricsCollector
    collector = DefaultMetricsCollector()
    collector.record_comparison()
    collector.record_swap()
    collector.record_access()

    with collector.timing():
        pass  # Quick operation

    result = collector.get_metrics()
    assert result.comparisons == 1
    assert result.swaps == 1
    assert result.accesses == 1
    assert result.execution_time > 0

    print("   ✓ Metrics collection working correctly")
except Exception as e:
    print(f"   ✗ Metrics tests failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test ThreadSafeMetricsCollector
print("\n5. Testing thread-safe metrics collector...")
try:
    collector = ThreadSafeMetricsCollector()
    collector.record_comparison()
    collector.record_swap()

    with collector.timing():
        collector.record_access()

    result = collector.get_metrics()
    assert result.comparisons == 1
    assert result.swaps == 1
    assert result.accesses == 1

    print("   ✓ Thread-safe collector working correctly")
except Exception as e:
    print(f"   ✗ Thread-safe collector tests failed: {e}")
    sys.exit(1)

# Test 6: Test protocol compliance
print("\n6. Testing MetricsCollector protocol compliance...")
try:
    from complexity_profiler.algorithms.base import MetricsCollector

    # Test that both collectors implement the protocol
    default_collector = DefaultMetricsCollector()
    thread_safe_collector = ThreadSafeMetricsCollector()

    # Both should have all required protocol methods
    for method in ['record_comparison', 'record_swap', 'record_access',
                   'record_recursive_call', 'get_metrics', 'reset']:
        assert hasattr(default_collector, method)
        assert hasattr(thread_safe_collector, method)

    print("   ✓ Protocol compliance verified")
except Exception as e:
    print(f"   ✗ Protocol compliance tests failed: {e}")
    sys.exit(1)

# Test 7: Test AggregateMetricsCollector
print("\n7. Testing aggregate metrics collector...")
try:
    aggregate = AggregateMetricsCollector()

    for i in range(3):
        collector = DefaultMetricsCollector()
        for _ in range(i + 1):
            collector.record_comparison()
        aggregate.add_metrics(collector.get_metrics())

    summary = aggregate.get_summary()
    assert summary['runs'] == 3
    assert summary['mean_comparisons'] == 2.0  # (1 + 2 + 3) / 3

    print("   ✓ Aggregate collector working correctly")
except Exception as e:
    print(f"   ✗ Aggregate collector tests failed: {e}")
    sys.exit(1)

# Test 8: Test measure_time utility
print("\n8. Testing measure_time utility...")
try:
    import time

    elapsed = []
    with measure_time() as elapsed:
        time.sleep(0.01)  # Sleep for 10ms

    assert elapsed[0] >= 0.01
    print(f"   ✓ measure_time utility working correctly (measured: {elapsed[0]:.4f}s)")
except Exception as e:
    print(f"   ✗ measure_time tests failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("All Phase 1 tests passed successfully!")
print("=" * 70)
print("\nImplemented components:")
print("  • Custom exception hierarchy (7 exception classes)")
print("  • Logging configuration with rich support")
print("  • Metrics collection system with thread-safety")
print("  • Full MetricsCollector protocol compliance")
print("=" * 70)
