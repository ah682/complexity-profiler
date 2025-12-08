"""
Custom exception hierarchy for the Big-O Complexity Analyzer.

This module defines a comprehensive exception hierarchy for handling various
error conditions that may arise during algorithm analysis, profiling, data
generation, and result export operations.

All exceptions inherit from BigOComplexityError, making it easy to catch
any library-specific errors with a single except clause.
"""

from typing import Optional, Any


class BigOComplexityError(Exception):
    """
    Base exception for all Big-O Complexity Analyzer errors.

    This is the root of the exception hierarchy. Catching this exception
    will catch all library-specific errors.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary containing additional error context
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class AlgorithmError(BigOComplexityError):
    """
    Exception raised when an algorithm execution fails.

    This exception is raised when an algorithm encounters an error during
    execution, such as invalid input data, internal logic errors, or
    unexpected conditions.

    Examples:
        >>> raise AlgorithmError(
        ...     "Sorting algorithm failed",
        ...     details={"algorithm": "quicksort", "input_size": 1000}
        ... )
    """

    def __init__(
        self,
        message: str,
        algorithm_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the algorithm error.

        Args:
            message: Description of what went wrong
            algorithm_name: Name of the algorithm that failed
            details: Additional context about the failure
        """
        details = details or {}
        if algorithm_name:
            details["algorithm"] = algorithm_name
        super().__init__(message, details)
        self.algorithm_name = algorithm_name


class DataGenerationError(BigOComplexityError):
    """
    Exception raised when test data generation fails.

    This exception is raised when the system fails to generate appropriate
    test data for algorithm profiling, such as when requested data
    characteristics are impossible or when random generation fails.

    Examples:
        >>> raise DataGenerationError(
        ...     "Cannot generate sorted array with negative size",
        ...     details={"requested_size": -100, "data_type": "sorted"}
        ... )
    """

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        size: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the data generation error.

        Args:
            message: Description of the generation failure
            data_type: Type of data that failed to generate (e.g., "sorted", "random")
            size: Size of data that was being generated
            details: Additional context about the failure
        """
        details = details or {}
        if data_type:
            details["data_type"] = data_type
        if size is not None:
            details["size"] = size
        super().__init__(message, details)
        self.data_type = data_type
        self.size = size


class ProfilingError(BigOComplexityError):
    """
    Exception raised when algorithm profiling fails.

    This exception is raised when the profiling system encounters errors
    during the measurement and analysis of algorithm performance, such as
    timeout conditions, resource exhaustion, or measurement failures.

    Examples:
        >>> raise ProfilingError(
        ...     "Profiling timeout exceeded",
        ...     details={"timeout_seconds": 30, "completed_runs": 5, "total_runs": 10}
        ... )
    """

    def __init__(
        self,
        message: str,
        run_number: Optional[int] = None,
        input_size: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the profiling error.

        Args:
            message: Description of the profiling failure
            run_number: The run number when the error occurred
            input_size: Size of input being profiled when error occurred
            details: Additional context about the failure
        """
        details = details or {}
        if run_number is not None:
            details["run_number"] = run_number
        if input_size is not None:
            details["input_size"] = input_size
        super().__init__(message, details)
        self.run_number = run_number
        self.input_size = input_size


class CurveFittingError(BigOComplexityError):
    """
    Exception raised when curve fitting to complexity functions fails.

    This exception is raised when the statistical curve fitting process
    fails to converge, produces invalid results, or encounters numerical
    instability when trying to determine the Big-O complexity from
    empirical measurements.

    Examples:
        >>> raise CurveFittingError(
        ...     "Failed to fit O(n log n) curve",
        ...     details={"complexity_class": "O(n log n)", "r_squared": 0.42}
        ... )
    """

    def __init__(
        self,
        message: str,
        complexity_class: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the curve fitting error.

        Args:
            message: Description of the fitting failure
            complexity_class: The complexity class that failed to fit
            details: Additional context (e.g., R² value, convergence info)
        """
        details = details or {}
        if complexity_class:
            details["complexity_class"] = complexity_class
        super().__init__(message, details)
        self.complexity_class = complexity_class


class ConfigurationError(BigOComplexityError):
    """
    Exception raised when configuration is invalid or incomplete.

    This exception is raised when the system configuration contains invalid
    values, missing required settings, or incompatible options.

    Examples:
        >>> raise ConfigurationError(
        ...     "Invalid profiling configuration",
        ...     details={"invalid_field": "max_iterations", "value": -5}
        ... )
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the configuration error.

        Args:
            message: Description of the configuration problem
            config_key: The configuration key that is invalid
            invalid_value: The invalid value that was provided
            details: Additional context about the configuration error
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        if invalid_value is not None:
            details["invalid_value"] = invalid_value
        super().__init__(message, details)
        self.config_key = config_key
        self.invalid_value = invalid_value


class ExportError(BigOComplexityError):
    """
    Exception raised when result export operations fail.

    This exception is raised when the system fails to export analysis
    results to various formats (JSON, CSV, HTML, etc.) due to I/O errors,
    formatting issues, or invalid output specifications.

    Examples:
        >>> raise ExportError(
        ...     "Failed to write JSON report",
        ...     details={"format": "json", "path": "/invalid/path/report.json"}
        ... )
    """

    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the export error.

        Args:
            message: Description of the export failure
            export_format: Format that was being exported (e.g., "json", "csv", "html")
            file_path: Path where export was attempted
            details: Additional context about the export failure
        """
        details = details or {}
        if export_format:
            details["export_format"] = export_format
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, details)
        self.export_format = export_format
        self.file_path = file_path


__all__ = [
    "BigOComplexityError",
    "AlgorithmError",
    "DataGenerationError",
    "ProfilingError",
    "CurveFittingError",
    "ConfigurationError",
    "ExportError",
]
