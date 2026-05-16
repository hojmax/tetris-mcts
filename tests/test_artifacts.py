from __future__ import annotations

from pathlib import Path

import pytest

from tetris_bot.ml.artifacts import copy_model_artifact_bundle
from tetris_bot.ml.weights import split_model_paths


def _create_dummy_model_artifact_bundle(onnx_path: Path) -> None:
    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    onnx_path.write_bytes(b"onnx-main")
    conv_path.write_bytes(b"onnx-conv")
    heads_path.write_bytes(b"onnx-heads")
    fc_path.write_bytes(b"fc-bin")


def test_copy_model_artifact_bundle_allows_in_place_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_onnx = tmp_path / "incumbent.onnx"
    _create_dummy_model_artifact_bundle(source_onnx)
    source_conv, source_heads, source_fc = split_model_paths(source_onnx)
    source_main_data = source_onnx.with_suffix(".onnx.data")
    source_conv_data = source_conv.with_suffix(".onnx.data")
    source_heads_data = source_heads.with_suffix(".onnx.data")
    source_main_data.write_bytes(b"main-data")
    source_conv_data.write_bytes(b"conv-data")
    source_heads_data.write_bytes(b"heads-data")

    monkeypatch.setattr(
        "tetris_bot.ml.artifacts._fix_onnx_external_data_references",
        lambda _path: None,
    )

    before_bytes = {
        path: path.read_bytes()
        for path in [
            source_onnx,
            source_conv,
            source_heads,
            source_fc,
            source_main_data,
            source_conv_data,
            source_heads_data,
        ]
    }
    copy_model_artifact_bundle(source_onnx, source_onnx)
    after_bytes = {path: path.read_bytes() for path in before_bytes}

    assert after_bytes == before_bytes
