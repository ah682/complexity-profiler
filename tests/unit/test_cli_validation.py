"""Validate CLI boundaries with controlled values, files and environment."""

import io
import logging

import click
import pytest

from complexity_profiler.cli import validators as v
from complexity_profiler.config.settings import Settings
from complexity_profiler.utils.logging_config import get_logger, setup_logging


def test_sizes_parser_handles_both_notations_and_sorting():
    assert v.validate_input_sizes(None, None, "100-1000:5") == [100, 325, 550, 775, 1000]
    assert v.validate_input_sizes(None, None, "8,2,4") == [2, 4, 8]
    assert v.validate_input_sizes(None, None, "32") == [32]


@pytest.mark.parametrize("value", ["", "x", "0", "1,-2,3", "0-4:3", "4-2:3", "2-4:1", "a-b:3"])
def test_sizes_rejects_invalid_values(value):
    with pytest.raises(click.BadParameter):
        v.validate_input_sizes(None, None, value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("validate_algorithm_name", "merge_sort"),
        ("validate_data_type", "random"),
        ("validate_export_format", "json"),
        ("validate_metric", "execution_time"),
    ],
)
def test_enum_validators_accept_supported_and_reject_unknown(name, value):
    validate = getattr(v, name)
    assert validate(None, None, value) == value
    with pytest.raises(click.BadParameter):
        validate(None, None, "unknown")


def test_multiple_algorithms_and_run_limits():
    assert v.validate_algorithm_names(None, None, ("merge_sort", "heap_sort")) == (
        "merge_sort",
        "heap_sort",
    )
    assert v.validate_runs(None, None, 1) == 1
    assert v.validate_runs(None, None, 100) == 100
    for value in [0, -1, 101]:
        with pytest.raises(click.BadParameter):
            v.validate_runs(None, None, value)


def test_output_validation_checks_parent_and_overwrite(monkeypatch, tmp_path):
    assert v.validate_output_path(None, None, None) is None
    path = tmp_path / "result.json"
    assert v.validate_output_path(None, None, str(path)) == path
    with pytest.raises(click.BadParameter):
        v.validate_output_path(None, None, str(tmp_path / "missing" / "report"))
    path.write_text("old")
    monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: False)
    with pytest.raises(click.Abort):
        v.validate_output_path(None, None, str(path))
    monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: True)
    assert v.validate_output_path(None, None, str(path)) == path


def test_algorithm_factory_executes_registered_algorithm():
    from complexity_profiler.analysis.metrics import DefaultMetricsCollector

    for name in v.AVAILABLE_ALGORITHMS:
        result = v.get_algorithm_instance(name).execute([3, 1, 2], DefaultMetricsCollector())
        assert result == [1, 2, 3]
    with pytest.raises(ValueError, match="Unknown"):
        v.get_algorithm_instance("missing")


def test_generators_preserve_requested_pattern_and_seed():
    for kind in ["random", "sorted", "reversed", "nearly_sorted"]:
        first = v.create_data_generator(kind, seed=42)(50)
        second = v.create_data_generator(kind, seed=42)(50)
        assert first == second
        assert len(first) == 50
    assert v.create_data_generator("sorted")(3) == [0, 1, 2]
    assert v.create_data_generator("reversed")(3) == [3, 2, 1]
    assert sorted(v.create_data_generator("nearly_sorted", seed=1)(20)) == list(range(20))
    with pytest.raises(ValueError):
        v.create_data_generator("missing")


def test_settings_loads_file_and_environment_overrides(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[general]\ndefault_runs=4\n[logging]\nlevel="WARNING"\n')
    settings = Settings.load(path)
    assert settings.general.default_runs == 4
    assert settings.logging.level == "WARNING"
    monkeypatch.setenv("BIGOCOMPLEXITY_RUNS", "7")
    monkeypatch.setenv("BIGOCOMPLEXITY_LOG_LEVEL", "DEBUG")
    settings = Settings.load(path)
    assert settings.general.default_runs == 7
    assert settings.logging.level == "DEBUG"
    monkeypatch.chdir(tmp_path)
    assert Settings.load().general.default_runs == 7


def test_logging_writes_file_and_replaces_handlers(tmp_path):
    # Restore global logging handlers after exercising the real configuration.
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    try:
        path = tmp_path / "nested" / "profiler.log"
        setup_logging(level="DEBUG", log_file=str(path), enable_rich=False)
        get_logger("test").warning("visible measurement")
        for handler in root.handlers:
            handler.flush()
        assert "visible measurement" in path.read_text()
        stream = io.StringIO()
        with __import__("contextlib").redirect_stderr(stream):
            setup_logging(level="INFO", enable_rich=True)
            get_logger("test").debug("hidden debug message")
            get_logger("test").warning("visible warning")
        assert "visible warning" in stream.getvalue()
        assert "hidden debug message" not in stream.getvalue()
    finally:
        for handler in root.handlers[:]:
            if handler not in old_handlers:
                handler.close()
        root.handlers = old_handlers
        root.setLevel(old_level)
