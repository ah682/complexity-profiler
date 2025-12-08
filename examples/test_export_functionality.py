"""
Comprehensive test and validation script for export functionality.

This script demonstrates all export capabilities and validates the
implementation with various test cases.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.analysis.metrics import PerformanceMetrics
from complexity_profiler.analysis.statistics import Statistics, compute_statistics
from complexity_profiler.export import (
    export_to_json,
    load_from_json,
    export_to_csv,
    export_metrics_summary_csv,
    load_from_csv,
)
from complexity_profiler.utils.exceptions import ExportError


def create_test_profile_result(
    algorithm_name: str = "QuickSort",
    category: str = "sorting",
    num_sizes: int = 3,
    runs_per_size: int = 2,
) -> ProfileResult:
    """
    Create a test ProfileResult with synthetic data.

    Args:
        algorithm_name: Name of the algorithm
        category: Category of the algorithm
        num_sizes: Number of input sizes to test
        runs_per_size: Number of runs per input size

    Returns:
        Complete ProfileResult object
    """
    input_sizes = [100 * (2 ** i) for i in range(num_sizes)]

    metrics_per_size = {}
    statistics_per_size = {}

    for size in input_sizes:
        metrics_list = []
        execution_times = []

        for run in range(runs_per_size):
            metrics = PerformanceMetrics(
                comparisons=int(size * (run + 1) * 0.5),
                swaps=int(size * (run + 1) * 0.3),
                accesses=int(size * (run + 1) * 0.7),
                recursive_calls=int(size * 0.05 * (run + 1)),
                execution_time=size * 0.001 * (run + 1),
                memory_operations=int(size * 0.1 * (run + 1)),
            )
            metrics_list.append(metrics)
            execution_times.append(metrics.execution_time)

        metrics_per_size[size] = metrics_list
        statistics_per_size[size] = compute_statistics(execution_times)

    return ProfileResult(
        algorithm_name=algorithm_name,
        category=category,
        input_sizes=input_sizes,
        metrics_per_size=metrics_per_size,
        statistics_per_size=statistics_per_size,
        empirical_complexity="O(n log n)",
        r_squared=0.98,
        timestamp=datetime.now(),
        notes=f"Test profile for {algorithm_name}",
    )


def test_json_export(output_dir: Path) -> bool:
    """
    Test JSON export functionality.

    Args:
        output_dir: Directory to write test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING JSON EXPORT")
    print("=" * 60)

    try:
        result = create_test_profile_result()
        json_file = output_dir / "test_json_export.json"

        # Test pretty printing
        print("Testing pretty print export...", end=" ")
        export_to_json(result, json_file, pretty=True)
        assert json_file.exists(), "JSON file not created"
        print("PASS")

        # Verify JSON structure
        print("Verifying JSON structure...", end=" ")
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert data['algorithm_name'] == "QuickSort"
        assert data['empirical_complexity'] == "O(n log n)"
        assert len(data['input_sizes']) == 3
        print("PASS")

        # Test compact export
        print("Testing compact export...", end=" ")
        compact_file = output_dir / "test_json_compact.json"
        export_to_json(result, compact_file, pretty=False)
        assert compact_file.exists()
        # Compact should be smaller
        assert compact_file.stat().st_size <= json_file.stat().st_size
        print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_json_load(output_dir: Path) -> bool:
    """
    Test JSON loading functionality.

    Args:
        output_dir: Directory with test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING JSON LOAD")
    print("=" * 60)

    try:
        # Create and export
        original = create_test_profile_result()
        json_file = output_dir / "test_json_load.json"
        export_to_json(original, json_file)

        # Load back
        print("Loading JSON file...", end=" ")
        loaded = load_from_json(json_file)
        print("PASS")

        # Verify data integrity
        print("Verifying data integrity...", end=" ")
        assert loaded.algorithm_name == original.algorithm_name
        assert loaded.category == original.category
        assert loaded.empirical_complexity == original.empirical_complexity
        assert loaded.r_squared == original.r_squared
        assert loaded.input_sizes == original.input_sizes
        print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_csv_export(output_dir: Path) -> bool:
    """
    Test CSV export functionality.

    Args:
        output_dir: Directory to write test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING CSV EXPORT")
    print("=" * 60)

    try:
        result = create_test_profile_result()
        csv_file = output_dir / "test_csv_export.csv"

        # Test full export
        print("Testing full CSV export...", end=" ")
        export_to_csv(result, csv_file)
        assert csv_file.exists(), "CSV file not created"
        print("PASS")

        # Verify CSV structure
        print("Verifying CSV structure...", end=" ")
        with open(csv_file, 'r') as f:
            header = f.readline().strip()
        assert 'input_size' in header
        assert 'algorithm' in header
        assert 'mean_execution_time' in header
        print("PASS")

        # Test summary export
        print("Testing summary CSV export...", end=" ")
        summary_file = output_dir / "test_csv_summary.csv"
        export_metrics_summary_csv(result, summary_file)
        assert summary_file.exists()
        print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_csv_load(output_dir: Path) -> bool:
    """
    Test CSV loading functionality.

    Args:
        output_dir: Directory with test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING CSV LOAD")
    print("=" * 60)

    try:
        # Create and export
        original = create_test_profile_result()
        csv_file = output_dir / "test_csv_load.csv"
        export_to_csv(original, csv_file)

        # Load back
        print("Loading CSV file...", end=" ")
        df = load_from_csv(csv_file, "QuickSort", "sorting")
        print("PASS")

        # Verify data
        print("Verifying CSV data...", end=" ")
        assert len(df) == len(original.input_sizes)
        assert list(df['input_size'].values) == original.input_sizes
        print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_error_handling(output_dir: Path) -> bool:
    """
    Test error handling.

    Args:
        output_dir: Directory for test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)

    try:
        # Test file not found
        print("Testing file not found error...", end=" ")
        try:
            load_from_json(Path("/nonexistent/file.json"))
            print("FAIL: Should have raised ExportError")
            return False
        except ExportError as e:
            assert "not found" in str(e).lower()
            print("PASS")

        # Test invalid path
        print("Testing invalid path error...", end=" ")
        result = create_test_profile_result()
        try:
            export_to_json(result, Path("/invalid/nonexistent/path/file.json"))
            print("FAIL: Should have raised ExportError")
            return False
        except ExportError as e:
            assert e.export_format == "json"
            print("PASS")

        # Test empty input sizes
        print("Testing validation error...", end=" ")
        try:
            invalid_result = ProfileResult(
                algorithm_name="Test",
                category="test",
                input_sizes=[],  # Invalid: empty
                metrics_per_size={},
                statistics_per_size={},
                empirical_complexity="O(n)",
                r_squared=0.95,
            )
            print("FAIL: Should have raised ValueError")
            return False
        except ValueError:
            print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_data_types(output_dir: Path) -> bool:
    """
    Test handling of various data types in JSON export.

    Args:
        output_dir: Directory for test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING DATA TYPE HANDLING")
    print("=" * 60)

    try:
        result = create_test_profile_result()
        json_file = output_dir / "test_datatypes.json"

        # Export
        print("Exporting with various data types...", end=" ")
        export_to_json(result, json_file)
        print("PASS")

        # Verify types in JSON
        print("Verifying JSON type conversion...", end=" ")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Check that numpy types were converted
        assert isinstance(data['r_squared'], float)
        assert isinstance(data['empirical_complexity'], str)
        assert isinstance(data['timestamp'], str)
        assert isinstance(data['input_sizes'], list)
        print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_profile_result_methods(output_dir: Path) -> bool:
    """
    Test ProfileResult methods.

    Args:
        output_dir: Directory for test files

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("TESTING PROFILERESULT METHODS")
    print("=" * 60)

    try:
        result = create_test_profile_result()

        # Test to_dict
        print("Testing to_dict()...", end=" ")
        data = result.to_dict()
        assert isinstance(data, dict)
        assert 'algorithm_name' in data
        assert 'metrics_per_size' in data
        print("PASS")

        # Test get_summary
        print("Testing get_summary()...", end=" ")
        summary = result.get_summary()
        assert summary['algorithm_name'] == result.algorithm_name
        assert summary['empirical_complexity'] == result.empirical_complexity
        assert summary['r_squared'] == result.r_squared
        print("PASS")

        # Test statistics methods
        print("Testing statistics methods...", end=" ")
        for size in result.input_sizes:
            stats = result.statistics_per_size[size]
            # Test is_consistent method
            is_consistent = stats.is_consistent(threshold=0.5)
            assert isinstance(is_consistent, bool)
        print("PASS")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def run_all_tests() -> bool:
    """
    Run all tests and print summary.

    Returns:
        True if all tests pass
    """
    # Create test output directory
    output_dir = Path("test_export_output")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPORT FUNCTIONALITY TEST SUITE")
    print("=" * 60)

    tests = [
        ("JSON Export", lambda: test_json_export(output_dir)),
        ("JSON Load", lambda: test_json_load(output_dir)),
        ("CSV Export", lambda: test_csv_export(output_dir)),
        ("CSV Load", lambda: test_csv_load(output_dir)),
        ("Error Handling", lambda: test_error_handling(output_dir)),
        ("Data Types", lambda: test_data_types(output_dir)),
        ("ProfileResult Methods", lambda: test_profile_result_methods(output_dir)),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name:.<40} {status}")

    print("=" * 60)
    print(f"Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nAll tests passed! Export functionality is working correctly.")
        return True
    else:
        print(f"\n{total_count - passed_count} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
