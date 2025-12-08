"""Default configuration values for Big-O Complexity Analyzer."""

# Default input sizes for profiling
DEFAULT_SIZES = [100, 500, 1000, 5000, 10000]

# Default number of runs per size
DEFAULT_RUNS = 10

# Default warmup runs
DEFAULT_WARMUP = 2

# Timeout for profiling (seconds)
DEFAULT_TIMEOUT = 300

# Visualization defaults
DEFAULT_DPI = 300
DEFAULT_CHART_FORMAT = "png"
DEFAULT_THEME = "seaborn-v0_8-darkgrid"

# Export defaults
DEFAULT_EXPORT_FORMAT = "json"

# Logging defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "complexity_profiler.log"

__all__ = [
    "DEFAULT_SIZES",
    "DEFAULT_RUNS",
    "DEFAULT_WARMUP",
    "DEFAULT_TIMEOUT",
    "DEFAULT_DPI",
    "DEFAULT_CHART_FORMAT",
    "DEFAULT_THEME",
    "DEFAULT_EXPORT_FORMAT",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FILE",
]
