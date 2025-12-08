"""
JSON export functionality for profiling results.

This module provides utilities for exporting algorithm profiling results
to JSON format with support for pretty printing and proper handling of
numpy and datetime types.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from complexity_profiler.analysis.profiler import ProfileResult
from complexity_profiler.utils.exceptions import ExportError


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy types.

    This encoder extends json.JSONEncoder to properly serialize numpy
    data types which are not natively JSON serializable.

    Example:
        >>> import numpy as np
        >>> data = {"value": np.int64(42), "array": np.array([1, 2, 3])}
        >>> json_str = json.dumps(data, cls=NumpyEncoder)
    """

    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-serializable types.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable representation of the object

        Raises:
            TypeError: If object type is not supported
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)


def export_to_json(
    result: ProfileResult,
    file_path: Path,
    pretty: bool = True,
) -> None:
    """
    Export profiling results to a JSON file.

    Converts a ProfileResult object to JSON format and writes it to the
    specified file path. Supports pretty printing for readability and
    handles numpy types and datetime objects automatically.

    Args:
        result: ProfileResult object containing the profiling data
        file_path: Path where the JSON file should be written
        pretty: If True, format JSON with indentation and sorting (default: True)

    Raises:
        ExportError: If the export operation fails due to I/O errors or
                    invalid file paths

    Example:
        >>> from complexity_profiler.analysis.profiler import ProfileResult
        >>> from pathlib import Path
        >>> result = ProfileResult(...)
        >>> export_to_json(result, Path("results.json"))
    """
    try:
        # Ensure parent directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert ProfileResult to dictionary
        data = result.to_dict()

        # Write to JSON file with optional pretty printing
        with open(file_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(
                    data,
                    f,
                    cls=NumpyEncoder,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            else:
                json.dump(
                    data,
                    f,
                    cls=NumpyEncoder,
                    ensure_ascii=False,
                    separators=(',', ':'),
                )

    except TypeError as e:
        raise ExportError(
            f"Failed to serialize data to JSON: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": "serialization"},
        ) from e
    except (IOError, OSError) as e:
        raise ExportError(
            f"Failed to write JSON file: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": "io_error"},
        ) from e
    except Exception as e:
        raise ExportError(
            f"Unexpected error during JSON export: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": type(e).__name__},
        ) from e


def load_from_json(file_path: Path) -> ProfileResult:
    """
    Load profiling results from a JSON file.

    Reads a JSON file created by export_to_json and reconstructs the
    ProfileResult object. Handles datetime and nested data structures.

    Args:
        file_path: Path to the JSON file to load

    Returns:
        Reconstructed ProfileResult object

    Raises:
        ExportError: If the file cannot be read or is invalid JSON

    Example:
        >>> from pathlib import Path
        >>> result = load_from_json(Path("results.json"))
    """
    try:
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct ProfileResult from dictionary
        # This is a placeholder - full reconstruction logic would be needed
        # based on the actual data structure
        return ProfileResult(
            algorithm_name=data['algorithm_name'],
            category=data['category'],
            input_sizes=data['input_sizes'],
            metrics_per_size=data['metrics_per_size'],
            statistics_per_size=data['statistics_per_size'],
            empirical_complexity=data['empirical_complexity'],
            r_squared=data['r_squared'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            notes=data.get('notes'),
        )

    except FileNotFoundError as e:
        raise ExportError(
            f"JSON file not found: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": "file_not_found"},
        ) from e
    except json.JSONDecodeError as e:
        raise ExportError(
            f"Invalid JSON file: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": "invalid_json"},
        ) from e
    except KeyError as e:
        raise ExportError(
            f"Missing required field in JSON: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": "missing_field", "field": str(e)},
        ) from e
    except Exception as e:
        raise ExportError(
            f"Unexpected error during JSON load: {str(e)}",
            export_format="json",
            file_path=str(file_path),
            details={"error_type": type(e).__name__},
        ) from e


__all__ = [
    "export_to_json",
    "load_from_json",
    "NumpyEncoder",
]
