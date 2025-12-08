"""
Logging configuration for the Big-O Complexity Analyzer.

This module provides a centralized logging configuration system with support
for both console output (using rich for beautiful formatting) and file logging
with automatic rotation. It follows Python logging best practices and provides
a simple setup_logging() function for initialization.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Default log format for file logging
DEFAULT_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)

# Simplified format for console logging (rich handles most formatting)
DEFAULT_CONSOLE_FORMAT = "%(message)s"

# Default date format
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
    log_file_level: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    enable_rich: bool = True,
    show_time: bool = True,
    show_level: bool = True,
    show_path: bool = True,
    tracebacks_show_locals: bool = False,
) -> logging.Logger:
    """
    Configure logging for the Big-O Complexity Analyzer.

    This function sets up a comprehensive logging system with:
    - Beautiful console output using rich (if available and enabled)
    - Optional file logging with automatic rotation
    - Proper formatting and level configuration
    - Enhanced traceback display for debugging

    Args:
        level: Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, file logging is disabled
        log_file_level: Log level for file output. If None, uses same as console
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup log files to keep
        enable_rich: Use rich for beautiful console output (if available)
        show_time: Show timestamp in console output (rich only)
        show_level: Show log level in console output (rich only)
        show_path: Show module path in console output (rich only)
        tracebacks_show_locals: Show local variables in tracebacks (rich only)

    Returns:
        Configured root logger instance

    Raises:
        ValueError: If an invalid log level is provided

    Examples:
        >>> # Basic setup with INFO level console logging
        >>> logger = setup_logging()

        >>> # Setup with DEBUG level and file logging
        >>> logger = setup_logging(
        ...     level="DEBUG",
        ...     log_file="logs/analyzer.log",
        ...     max_file_size=5_000_000,  # 5 MB
        ...     backup_count=3
        ... )

        >>> # Setup without rich formatting
        >>> logger = setup_logging(enable_rich=False)
    """
    # Validate and convert log level
    numeric_level = _get_numeric_level(level)
    file_numeric_level = (
        _get_numeric_level(log_file_level) if log_file_level else numeric_level
    )

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG, handlers will filter
    root_logger.handlers.clear()

    # Setup console handler
    console_handler = _create_console_handler(
        level=numeric_level,
        enable_rich=enable_rich and RICH_AVAILABLE,
        show_time=show_time,
        show_level=show_level,
        show_path=show_path,
    )
    root_logger.addHandler(console_handler)

    # Setup file handler if requested
    if log_file:
        file_handler = _create_file_handler(
            log_file=log_file,
            level=file_numeric_level,
            max_file_size=max_file_size,
            backup_count=backup_count,
        )
        root_logger.addHandler(file_handler)

    # Install rich traceback handler if available and enabled
    if enable_rich and RICH_AVAILABLE:
        install_rich_traceback(
            show_locals=tracebacks_show_locals,
            suppress=[],  # Don't suppress any modules
            max_frames=20,
        )

    # Log initial configuration
    root_logger.debug(
        "Logging configured: console_level=%s, file_level=%s, file=%s, rich=%s",
        level,
        log_file_level or level,
        log_file or "disabled",
        enable_rich and RICH_AVAILABLE,
    )

    return root_logger


def _get_numeric_level(level: str) -> int:
    """
    Convert string log level to numeric value.

    Args:
        level: String log level (case-insensitive)

    Returns:
        Numeric log level

    Raises:
        ValueError: If level is invalid
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        raise ValueError(
            f"Invalid log level: {level}. Must be one of {valid_levels}"
        )
    return numeric_level


def _create_console_handler(
    level: int,
    enable_rich: bool,
    show_time: bool,
    show_level: bool,
    show_path: bool,
) -> logging.Handler:
    """
    Create and configure console logging handler.

    Args:
        level: Numeric log level
        enable_rich: Whether to use rich handler
        show_time: Show timestamps (rich only)
        show_level: Show log levels (rich only)
        show_path: Show module paths (rich only)

    Returns:
        Configured console handler
    """
    if enable_rich:
        # Create rich console with error output to stderr
        console = Console(file=sys.stderr, force_terminal=True)

        # Create rich handler with configuration
        handler = RichHandler(
            level=level,
            console=console,
            show_time=show_time,
            show_level=show_level,
            show_path=show_path,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            tracebacks_suppress=[],
        )
        handler.setFormatter(logging.Formatter(DEFAULT_CONSOLE_FORMAT))
    else:
        # Fallback to standard stream handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt=DEFAULT_FILE_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)

    return handler


def _create_file_handler(
    log_file: Path | str,
    level: int,
    max_file_size: int,
    backup_count: int,
) -> RotatingFileHandler:
    """
    Create and configure rotating file logging handler.

    Args:
        log_file: Path to log file
        level: Numeric log level
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured rotating file handler
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)

    # Set formatter
    formatter = logging.Formatter(
        fmt=DEFAULT_FILE_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This is a convenience function that returns a properly configured
    logger for the given module name. It should be called at module level
    with __name__ as the argument.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Examples:
        >>> # At module level
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting algorithm analysis")
    """
    return logging.getLogger(name)


def set_module_level(module_name: str, level: str) -> None:
    """
    Set log level for a specific module or package.

    This allows fine-grained control over logging verbosity for different
    parts of the application.

    Args:
        module_name: Name of the module or package
        level: Log level to set (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples:
        >>> # Reduce verbosity for matplotlib
        >>> set_module_level("matplotlib", "WARNING")

        >>> # Enable debug logging for specific module
        >>> set_module_level("complexity_profiler.analysis", "DEBUG")
    """
    numeric_level = _get_numeric_level(level)
    logging.getLogger(module_name).setLevel(numeric_level)


__all__ = [
    "setup_logging",
    "get_logger",
    "set_module_level",
]
