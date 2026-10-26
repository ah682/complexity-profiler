"""Consumer-facing exports, plots, displays and CLI behavior on literal data."""

import csv
import io
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner
from rich.console import Console

from complexity_profiler.analysis.metrics import PerformanceMetrics
from complexity_profiler.analysis.profiler import AlgorithmProfiler, ProfileResult
from complexity_profiler.analysis.statistics import compute_statistics
from complexity_profiler.cli.main import cli
from complexity_profiler.export import (
    export_metrics_summary_csv,
    export_to_csv,
    export_to_json,
    load_from_csv,
    load_from_json,
)
from complexity_profiler.export.json_exporter import NumpyEncoder
from complexity_profiler.utils.exceptions import ExportError
from complexity_profiler.visualization.charts import (
    ChartGenerator,
    plot_comparison,
    plot_complexity_curve,
)
from complexity_profiler.visualization.formatters import (
    ResultFormatter,
    format_metrics,
    format_profile_result,
    format_results_table,
)


@pytest.fixture
def observations():
    metrics = {
        n: [PerformanceMetrics(execution_time=t, comparisons=n, swaps=2, accesses=3)]
        for n, t in [(2, 0.002), (4, 0.004), (8, 0.008)]
    }
    return ProfileResult(
        "Merge Sort",
        "sorting",
        [2, 4, 8],
        metrics,
        {
            n: compute_statistics([m.execution_time for m in values])
            for n, values in metrics.items()
        },
        "O(n)",
        1.0,
        timestamp=datetime(2026, 1, 2),
        notes="controlled data",
    )


def test_json_round_trip_and_numpy_encoding(observations, tmp_path):
    path = tmp_path / "nested" / "result.json"
    for pretty in [True, False]:
        export_to_json(observations, path, pretty=pretty)
        loaded = load_from_json(path)
        assert loaded.to_dict() == observations.to_dict()
    encoded = json.loads(
        json.dumps(
            {
                "integer": np.int64(4),
                "float": np.float64(0.25),
                "array": np.array([1, 2]),
                "date": datetime(2026, 1, 2),
            },
            cls=NumpyEncoder,
        )
    )
    assert encoded == {"integer": 4, "float": 0.25, "array": [1, 2], "date": "2026-01-02T00:00:00"}
    with pytest.raises(TypeError):
        json.dumps(object(), cls=NumpyEncoder)


@pytest.mark.parametrize("payload", ["{", "{}"])
def test_json_rejects_malformed_reports(payload, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ExportError):
        load_from_json(path)


def test_export_errors_keep_context(observations, tmp_path):
    for loader in [load_from_json, lambda p: load_from_csv(p, "Merge Sort", "sorting")]:
        with pytest.raises(ExportError) as exc:
            loader(tmp_path / "missing")
        assert exc.value.details["error_type"] == "file_not_found"
    for exporter in [export_to_json, export_to_csv, export_metrics_summary_csv]:
        with pytest.raises(ExportError) as exc:
            exporter(observations, tmp_path)
        assert exc.value.details["error_type"] == "io_error"


def test_csv_aggregates_literal_observations(observations, tmp_path):
    path = tmp_path / "nested" / "result.csv"
    export_to_csv(observations, path)
    data = load_from_csv(path, "Merge Sort", "sorting")
    assert data["input_size"].tolist() == [2, 4, 8]
    assert data["mean_comparisons"].tolist() == [2, 4, 8]
    assert data["mean_execution_time"].tolist() == [0.002, 0.004, 0.008]
    assert data["mean_total_operations"].tolist() == [7, 9, 13]
    with pytest.raises(ExportError, match="does not match"):
        load_from_csv(path, "Quick Sort", "sorting")
    summary = tmp_path / "summary.csv"
    export_metrics_summary_csv(observations, summary)
    row = next(csv.DictReader(summary.open()))
    assert row["empirical_complexity"] == "O(n)"
    assert row["input_sizes"] == "2,4,8"
    observations.metrics_per_size = {}
    with pytest.raises(ExportError, match="No data"):
        export_to_csv(observations, path)


@pytest.mark.parametrize(
    "metric,want",
    [
        ("execution_time", [0.002, 0.004, 0.008]),
        ("comparisons", [2, 4, 8]),
        ("swaps", [2, 2, 2]),
        ("accesses", [3, 3, 3]),
        ("total_operations", [7, 9, 13]),
    ],
)
def test_chart_coordinates_are_measurements(observations, metric, want, tmp_path):
    generator = ChartGenerator()
    fig = generator.plot_complexity_curve(
        observations,
        metric=metric,
        show_error_bars=False,
        log_scale=True,
        save_path=tmp_path / "plot.png",
    )
    assert fig.axes[0].lines[0].get_ydata().tolist() == want
    assert fig.axes[0].get_xscale() == "log"
    assert (tmp_path / "plot.png").read_bytes().startswith(b"\x89PNG")
    generator.close()
    assert generator.current_figure is None


def test_chart_comparison_overview_and_validation(observations, tmp_path):
    generator = ChartGenerator()
    with pytest.raises(ValueError, match="No figure"):
        generator.save(tmp_path / "bad.png")
    with pytest.raises(ValueError, match="empty"):
        generator.plot_comparison([])
    with pytest.raises(ValueError, match="Invalid metric"):
        generator.plot_complexity_curve(observations, metric="bad")
    fig = generator.plot_comparison(
        [observations, observations], log_scale=True, save_path=tmp_path / "compare.png"
    )
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].lines[0].get_xdata().tolist() == [2, 4, 8]
    generator.close()
    fig = generator.plot_metrics_overview(observations, save_path=tmp_path / "overview.png")
    assert len(fig.axes) == 4
    assert fig.axes[1].lines[0].get_ydata().tolist() == [2, 4, 8]
    generator.close()
    assert plot_complexity_curve(observations).axes
    assert plot_comparison([observations]).axes
    plt.close("all")


def test_console_outputs_include_observed_data_and_class(observations):
    stream = io.StringIO()
    console = Console(file=stream, width=140, color_system=None)
    formatter = ResultFormatter(console)
    renderables = [
        format_profile_result(observations),
        format_results_table(observations),
        formatter.format_comparison_table([observations]),
        formatter.format_statistics(observations.statistics_per_size[2]),
        formatter.format_tree_view([observations]),
    ]
    for item in renderables:
        console.print(item)
    console.print(format_metrics(observations.metrics_per_size[2][0]))
    for name in ["print_success", "print_error", "print_warning", "print_info"]:
        getattr(formatter, name)("test message")
    output = stream.getvalue()
    assert "Merge Sort" in output and "O(n)" in output
    assert "0.002000" in output and "Comparisons" in output
    assert "2, 4, 8" in output and "test message" in output
    assert formatter._format_list([1, 2, 3, 4], 2) == "1, 2, ... (4 total)"


@pytest.mark.parametrize("export", ["json", "csv"])
def test_cli_analyze_exports_a_real_report(export, tmp_path):
    path = tmp_path / ("report." + export)
    result = CliRunner().invoke(
        cli,
        [
            "analyze",
            "merge_sort",
            "--sizes",
            "16,32,64",
            "--runs",
            "2",
            "--no-visualize",
            "--export",
            export,
            "--output",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert path.exists()
    if export == "json":
        assert load_from_json(path).input_sizes == [16, 32, 64]
    else:
        assert load_from_csv(path, "Merge Sort", "sorting")["num_runs"].tolist() == [2, 2, 2]


def test_cli_compare_uses_profile_fields(observations, monkeypatch, tmp_path):
    monkeypatch.setattr(AlgorithmProfiler, "profile", lambda self, algorithm: observations)
    path = tmp_path / "compare.png"
    result = CliRunner().invoke(
        cli, ["compare", "merge_sort", "quick_sort", "--save-chart", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert "Merge Sort" in result.output and "1.0000" in result.output
    assert path.exists()
    plt.close("all")


def test_cli_help_listing_and_invalid_arguments():
    runner = CliRunner()
    for args in [
        ["--help"],
        ["--version"],
        ["list-algorithms"],
        ["list-algorithms", "--list-searching"],
    ]:
        assert runner.invoke(cli, args).exit_code == 0
    for args in [
        ["analyze", "bad"],
        ["analyze", "merge_sort", "--sizes", "bad", "--no-visualize"],
        ["compare", "missing"],
    ]:
        assert runner.invoke(cli, args).exit_code != 0


@pytest.mark.parametrize("format", ["json", "csv", "html"])
def test_extended_analysis_command_writes_reports(format, tmp_path):
    from complexity_profiler.cli.commands.analyze import analyze

    path = tmp_path / ("report." + format)
    chart = tmp_path / "plot.png"
    result = CliRunner().invoke(
        analyze,
        [
            "merge_sort",
            "--sizes",
            "16,32,64",
            "--runs",
            "2",
            "--export",
            format,
            "--output",
            str(path),
            "--visualize",
            str(chart),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert path.stat().st_size > 0
    assert chart.read_bytes().startswith(b"\x89PNG")
    if format == "json":
        assert json.loads(path.read_text())["input_sizes"] == [16, 32, 64]
    elif format == "html":
        assert "Merge Sort" in path.read_text()
    else:
        assert len(list(csv.DictReader(path.open()))) == 3
    plt.close("all")


def test_inconclusive_console_shows_reason_and_candidate_evidence(observations):
    observations.empirical_complexity = "Inconclusive"
    observations.notes = "ambiguous candidate scores"
    observations.candidate_scores = {"O(n)": 0.991, "O(n log n)": 0.990}
    stream = io.StringIO()
    Console(file=stream, width=140, color_system=None).print(format_profile_result(observations))
    assert "ambiguous candidate scores" in stream.getvalue()
    assert "0.9910" in stream.getvalue()


def test_chart_calls_measured_trend_a_trend(observations):
    generator = ChartGenerator()
    figure = generator.plot_complexity_curve(observations, show_error_bars=False)
    assert "Mean trend" in figure.axes[0].get_legend_handles_labels()[1]
    generator.close()
