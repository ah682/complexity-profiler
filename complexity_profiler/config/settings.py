"""Configuration management for Big-O Complexity Analyzer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


@dataclass
class GeneralSettings:
    """General application settings."""

    default_runs: int = 10
    default_sizes: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])
    random_seed: Optional[int] = None


@dataclass
class ProfilingSettings:
    """Profiling-specific settings."""

    warmup_runs: int = 2
    timeout_seconds: int = 300
    min_run_time: float = 0.001


@dataclass
class VisualizationSettings:
    """Visualization settings."""

    theme: str = "seaborn-v0_8-darkgrid"
    default_format: str = "png"
    dpi: int = 300
    show_confidence_intervals: bool = True
    confidence_level: float = 0.95


@dataclass
class ExportSettings:
    """Export settings."""

    default_format: str = "json"
    pretty_print: bool = True
    include_raw_data: bool = False


@dataclass
class LoggingSettings:
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "complexity_profiler.log"


@dataclass
class Settings:
    """
    Application settings with environment variable overrides.

    Loads from TOML configuration file with support for environment
    variable overrides.
    """

    general: GeneralSettings
    profiling: ProfilingSettings
    visualization: VisualizationSettings
    export: ExportSettings
    logging: LoggingSettings

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """
        Load settings from file with environment overrides.

        Args:
            config_path: Path to TOML config file. If None, uses default locations.

        Returns:
            Settings object with loaded configuration

        Raises:
            ConfigurationError: If config file is invalid
        """
        # Default config paths
        if config_path is None:
            # Try current directory first, then user config
            candidates = [
                Path("config.toml"),
                Path.home() / ".config" / "complexity_profiler" / "config.toml",
            ]
            config_path = next((p for p in candidates if p.exists()), None)

        # Load from file if exists
        config = {}
        if config_path and config_path.exists():
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

        # Apply environment variable overrides
        if runs := os.getenv("BIGOCOMPLEXITY_RUNS"):
            config.setdefault("general", {})["default_runs"] = int(runs)

        if level := os.getenv("BIGOCOMPLEXITY_LOG_LEVEL"):
            config.setdefault("logging", {})["level"] = level

        # Build settings object
        return cls(
            general=GeneralSettings(**config.get("general", {})),
            profiling=ProfilingSettings(**config.get("profiling", {})),
            visualization=VisualizationSettings(**config.get("visualization", {})),
            export=ExportSettings(**config.get("export", {})),
            logging=LoggingSettings(**config.get("logging", {})),
        )


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings object (singleton)
    """
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


__all__ = [
    "Settings",
    "GeneralSettings",
    "ProfilingSettings",
    "VisualizationSettings",
    "ExportSettings",
    "LoggingSettings",
    "get_settings",
]
