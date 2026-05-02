from __future__ import annotations

from pathlib import Path

import pytest

from tetris_bot.ml.weights import FC_BINARY_MAGIC, split_model_paths
from tetris_bot.scripts.inspection import optimize_machine


def _write_split_bundle(onnx_path: Path, *, fc_magic: bytes) -> None:
    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(b"onnx-main")
    conv_path.write_bytes(b"onnx-conv")
    heads_path.write_bytes(b"onnx-heads")
    fc_path.write_bytes(fc_magic + b"-payload")


def test_ensure_valid_args_rejects_incompatible_fc_binary(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    _write_split_bundle(model_path, fc_magic=b"\x80\x00\x00\x00")

    args = optimize_machine.ScriptArgs(
        model_path=model_path,
        worker_candidates=[2],
        compile_profiles=["tuned"],
        backends=["tract"],
        primary_backend="tract",
    )

    with pytest.raises(ValueError, match="expected magic TCM2"):
        optimize_machine.ensure_valid_args(args)


def test_resolve_model_path_skips_incompatible_benchmark_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "benchmarks" / "models" / "v3_latest.onnx"
    training_runs_dir = tmp_path / "training_runs"
    training_model_path = training_runs_dir / "v1" / "checkpoints" / "latest.onnx"
    auto_path = tmp_path / "benchmarks" / "models" / "optimize_bootstrap.onnx"

    _write_split_bundle(benchmark_path, fc_magic=b"\x80\x00\x00\x00")
    _write_split_bundle(training_model_path, fc_magic=FC_BINARY_MAGIC)

    monkeypatch.setattr(optimize_machine, "_BENCHMARK_MODEL_PATH", benchmark_path)
    monkeypatch.setattr(optimize_machine, "TRAINING_RUNS_DIR", training_runs_dir)
    monkeypatch.setattr(optimize_machine, "_AUTO_MODEL_PATH", auto_path)

    assert optimize_machine.resolve_model_path(None) == training_model_path


def test_resolve_model_path_generates_bootstrap_bundle_when_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "benchmarks" / "models" / "v3_latest.onnx"
    training_runs_dir = tmp_path / "training_runs"
    auto_path = tmp_path / "benchmarks" / "models" / "optimize_bootstrap.onnx"

    _write_split_bundle(benchmark_path, fc_magic=b"\x80\x00\x00\x00")

    monkeypatch.setattr(optimize_machine, "_BENCHMARK_MODEL_PATH", benchmark_path)
    monkeypatch.setattr(optimize_machine, "TRAINING_RUNS_DIR", training_runs_dir)
    monkeypatch.setattr(optimize_machine, "_AUTO_MODEL_PATH", auto_path)

    class DummyModel:
        def __init__(self, **_: object) -> None:
            pass

        def eval(self) -> DummyModel:
            return self

    def fake_export_onnx(_model: object, onnx_path: Path) -> bool:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(b"onnx-main")
        return True

    def fake_export_split_models(_model: object, onnx_path: Path) -> bool:
        _write_split_bundle(onnx_path, fc_magic=FC_BINARY_MAGIC)
        return True

    monkeypatch.setattr(optimize_machine, "TetrisNet", DummyModel)
    monkeypatch.setattr(optimize_machine, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        optimize_machine, "export_split_models", fake_export_split_models
    )

    assert optimize_machine.resolve_model_path(None) == auto_path
    assert optimize_machine.split_model_bundle_error(auto_path) is None
