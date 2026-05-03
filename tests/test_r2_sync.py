"""Tests for tetris_bot.ml.r2_sync.

These tests exercise the threading + cursor + key-layout logic with an
in-memory S3 mock and fake generator adapters; they do not require real R2
credentials or boto3 transport.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from tetris_bot.ml.r2_sync import (
    MACHINE_OFFSET_BLOCK_SIZE,
    ChunkDownloader,
    ChunkUploader,
    GameStatsDownloader,
    GameStatsUploader,
    MachineOffsetTable,
    ModelDownloader,
    R2Settings,
    download_model_bundle,
    fetch_model_pointer,
    upload_model_bundle,
)


# ---------------------------------------------------------------------------
# In-memory S3 mock
# ---------------------------------------------------------------------------


class _MockClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


@pytest.fixture(autouse=True)
def _patch_botocore_clienterror(monkeypatch):
    """Make `botocore.exceptions.ClientError` reachable from r2_sync.

    r2_sync imports ClientError lazily inside functions; we patch botocore so
    `isinstance(_MockClientError(...), ClientError)` works in those branches.
    """

    class _DummyBotocore:
        class exceptions:
            ClientError = _MockClientError

    import sys

    sys.modules.setdefault("botocore", _DummyBotocore)  # type: ignore[arg-type]
    sys.modules.setdefault(
        "botocore.exceptions",
        _DummyBotocore.exceptions,  # type: ignore[arg-type]
    )
    yield


class InMemoryS3:
    """Tiny in-memory S3 (per-bucket key-value store)."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def put_object(self, *, Bucket: str, Key: str, Body, **_: Any) -> dict[str, Any]:
        if isinstance(Body, (bytes, bytearray)):
            data = bytes(Body)
        elif hasattr(Body, "read"):
            data = Body.read()
        else:
            data = bytes(Body)
        with self._lock:
            self._objects[(Bucket, Key)] = data
        return {}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        with self._lock:
            data = self._objects.get((Bucket, Key))
        if data is None:
            raise _MockClientError("NoSuchKey")

        class _Body:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

        return {"Body": _Body(data)}

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        with self._lock:
            if (Bucket, Key) not in self._objects:
                raise _MockClientError("404")
        return {}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        Delimiter: str | None = None,
        StartAfter: str | None = None,
        ContinuationToken: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        with self._lock:
            keys = sorted(
                key
                for (bucket, key) in self._objects
                if bucket == Bucket and key.startswith(Prefix)
            )
        if StartAfter is not None:
            keys = [k for k in keys if k > StartAfter]
        contents: list[dict[str, Any]] = []
        common_prefixes: dict[str, None] = {}
        for key in keys:
            if Delimiter:
                rel = key[len(Prefix) :]
                if Delimiter in rel:
                    sub = Prefix + rel.split(Delimiter, 1)[0] + Delimiter
                    common_prefixes.setdefault(sub, None)
                    continue
            contents.append({"Key": key})
        return {
            "Contents": contents,
            "CommonPrefixes": [{"Prefix": p} for p in common_prefixes.keys()],
            "IsTruncated": False,
        }

    def upload_file(self, Filename: str, Bucket: str, Key: str, **_: Any) -> None:
        data = Path(Filename).read_bytes()
        with self._lock:
            self._objects[(Bucket, Key)] = data

    def download_file(self, Bucket: str, Key: str, Filename: str, **_: Any) -> None:
        with self._lock:
            data = self._objects.get((Bucket, Key))
        if data is None:
            raise _MockClientError("NoSuchKey")
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_bytes(data)


# ---------------------------------------------------------------------------
# Fake generator adapters
# ---------------------------------------------------------------------------


class FakeReplaySource:
    """Pretends to be a GameGenerator for upload-side tests.

    Each `dump_replay_delta_to_npz` call writes `count_per_call` placeholder
    bytes to disk and advances the logical window.
    """

    def __init__(self, total_examples: int, count_per_call: int) -> None:
        self._total = total_examples
        self._count_per_call = count_per_call
        self._produced = 0

    def buffer_size(self) -> int:
        return max(self._total - self._produced, 0)

    def dump_replay_delta_to_npz(
        self, filepath: str, from_index: int, max_examples: int
    ) -> tuple[int, int, int, int] | None:
        if self._total == 0:
            return None
        window_start = 0
        window_end = self._total
        slice_start = max(from_index, window_start)
        if slice_start >= window_end:
            return (window_start, window_end, slice_start, 0)
        n = min(self._count_per_call, max_examples, window_end - slice_start)
        Path(filepath).write_bytes(
            json.dumps({"slice_start": slice_start, "n": n}).encode("utf-8")
        )
        self._produced = max(self._produced, slice_start + n)
        return (window_start, window_end, slice_start, n)


class FakeReplaySink:
    """Pretends to be a GameGenerator for download-side tests."""

    def __init__(self) -> None:
        self.ingested: list[tuple[int, int, int]] = []
        self._lock = threading.Lock()

    def ingest_examples_from_npz(
        self, filepath: str, game_number_offset: int = 0
    ) -> int:
        payload = json.loads(Path(filepath).read_bytes())
        with self._lock:
            self.ingested.append(
                (
                    int(payload["slice_start"]),
                    int(payload["n"]),
                    int(game_number_offset),
                )
            )
        return int(payload["n"])


class FakeCompletedGamesSource:
    """Pretends to be a GameGenerator for game-stats upload tests."""

    def __init__(self, batches: list[list[dict]]) -> None:
        self._batches = list(batches)

    def drain_completed_games(self) -> list[dict]:
        if not self._batches:
            return []
        return self._batches.pop(0)


class FakeGameStatsSink:
    """Captures pushed game-stats batches with their applied offsets."""

    def __init__(self) -> None:
        self.batches: list[tuple[list[dict], int, str]] = []
        self._lock = threading.Lock()

    def push_remote_completed_games(
        self, entries: list[dict], game_number_offset: int, machine_id: str
    ) -> None:
        with self._lock:
            self.batches.append((entries, game_number_offset, machine_id))


class FakeModelSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []
        self._return_value = True

    def sync_model_directly(
        self, model_path: str, model_step: int, nn_value_weight: float
    ) -> bool:
        self.calls.append((model_path, model_step, nn_value_weight))
        return self._return_value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> R2Settings:
    return R2Settings(
        endpoint_url="https://mock.r2",
        bucket="bucket",
        prefix="tetris-mcts",
        sync_run_id="v9",
        access_key_id="key",
        secret_access_key="secret",
        request_timeout_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunk_uploader_uploads_and_advances_cursor(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    source = FakeReplaySource(total_examples=10, count_per_call=4)
    cursor = tmp_path / "upload_cursor.json"

    uploader = ChunkUploader(
        generator=source,
        settings=settings,
        machine_id="laptop",
        cursor_path=cursor,
        chunk_max_examples=4,
        upload_interval_seconds=10.0,
        client_factory=lambda: s3,
    )
    client = s3
    # Drive three iterations directly to avoid sleeping in the test thread.
    uploader._upload_one_chunk(client)  # type: ignore[arg-type]
    uploader._upload_one_chunk(client)  # type: ignore[arg-type]
    uploader._upload_one_chunk(client)  # type: ignore[arg-type]

    listing = s3.list_objects_v2(Bucket="bucket", Prefix=settings.replay_prefix())
    keys = [obj["Key"] for obj in listing["Contents"]]
    assert keys == [
        f"{settings.replay_machine_prefix('laptop')}{0:020d}.npz",
        f"{settings.replay_machine_prefix('laptop')}{4:020d}.npz",
        f"{settings.replay_machine_prefix('laptop')}{8:020d}.npz",
    ]
    assert json.loads(cursor.read_text())["next_from_index"] == 10


def test_chunk_uploader_skips_when_buffer_empty(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    source = FakeReplaySource(total_examples=0, count_per_call=4)

    uploader = ChunkUploader(
        generator=source,
        settings=settings,
        machine_id="laptop",
        cursor_path=tmp_path / "cursor.json",
        chunk_max_examples=4,
        upload_interval_seconds=10.0,
        client_factory=lambda: s3,
    )
    uploader._upload_one_chunk(s3)  # type: ignore[arg-type]
    listing = s3.list_objects_v2(Bucket="bucket", Prefix=settings.replay_prefix())
    assert listing["Contents"] == []


def test_chunk_downloader_ingests_new_keys(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    # Simulate two machines having uploaded a few chunks already.
    for slice_start in (0, 4, 8):
        key = settings.replay_chunk_key("laptop", slice_start)
        payload = json.dumps({"slice_start": slice_start, "n": 4}).encode()
        s3.put_object(Bucket="bucket", Key=key, Body=payload)
    for slice_start in (0, 4):
        key = settings.replay_chunk_key("desktop", slice_start)
        payload = json.dumps({"slice_start": slice_start, "n": 4}).encode()
        s3.put_object(Bucket="bucket", Key=key, Body=payload)

    sink = FakeReplaySink()
    cursor = tmp_path / "download_cursor.json"
    offset_table = MachineOffsetTable(tmp_path / "offsets.json")
    downloader = ChunkDownloader(
        generator=sink,
        settings=settings,
        cursor_path=cursor,
        poll_interval_seconds=10.0,
        offset_table=offset_table,
        client_factory=lambda: s3,
    )
    downloader._poll_once(s3)  # type: ignore[arg-type]

    laptop_offset = offset_table.offset_for("laptop")
    desktop_offset = offset_table.offset_for("desktop")
    assert sorted(sink.ingested) == sorted(
        [
            (0, 4, laptop_offset),
            (4, 4, laptop_offset),
            (8, 4, laptop_offset),
            (0, 4, desktop_offset),
            (4, 4, desktop_offset),
        ]
    )
    saved = json.loads(cursor.read_text())["per_machine"]
    assert saved["laptop"].endswith("00000000000000000008.npz")
    assert saved["desktop"].endswith("00000000000000000004.npz")

    # Adding a new chunk should pull only the new key on next poll.
    sink.ingested.clear()
    new_key = settings.replay_chunk_key("laptop", 12)
    s3.put_object(
        Bucket="bucket",
        Key=new_key,
        Body=json.dumps({"slice_start": 12, "n": 4}).encode(),
    )
    downloader._poll_once(s3)  # type: ignore[arg-type]
    assert sink.ingested == [(12, 4, laptop_offset)]


def test_uploader_then_downloader_round_trip(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    source = FakeReplaySource(total_examples=12, count_per_call=4)
    sink = FakeReplaySink()

    uploader = ChunkUploader(
        generator=source,
        settings=settings,
        machine_id="laptop",
        cursor_path=tmp_path / "u_cursor.json",
        chunk_max_examples=4,
        upload_interval_seconds=10.0,
        client_factory=lambda: s3,
    )
    offset_table = MachineOffsetTable(tmp_path / "offsets.json")
    downloader = ChunkDownloader(
        generator=sink,
        settings=settings,
        cursor_path=tmp_path / "d_cursor.json",
        poll_interval_seconds=10.0,
        offset_table=offset_table,
        client_factory=lambda: s3,
    )

    for _ in range(3):
        uploader._upload_one_chunk(s3)  # type: ignore[arg-type]
    downloader._poll_once(s3)  # type: ignore[arg-type]

    laptop_offset = offset_table.offset_for("laptop")
    assert sink.ingested == [
        (0, 4, laptop_offset),
        (4, 4, laptop_offset),
        (8, 4, laptop_offset),
    ]


def test_model_bundle_round_trip(tmp_path: Path, settings: R2Settings) -> None:
    s3 = InMemoryS3()
    bundle_dir = tmp_path / "bundle_src"
    bundle_dir.mkdir()
    main = bundle_dir / "latest.onnx"
    main.write_bytes(b"main-bytes")
    (bundle_dir / "latest.conv.onnx").write_bytes(b"conv-bytes")
    (bundle_dir / "latest.heads.onnx").write_bytes(b"heads-bytes")
    (bundle_dir / "latest.fc.bin").write_bytes(b"fc-bytes")

    pointer = upload_model_bundle(
        settings=settings,
        onnx_path=main,
        step=42,
        nn_value_weight=0.75,
        client=s3,
    )
    assert pointer.step == 42
    fetched = fetch_model_pointer(settings, client=s3)
    assert fetched == pointer

    dest_dir = tmp_path / "bundle_dst"
    main_path = download_model_bundle(
        settings=settings, pointer=pointer, dest_dir=dest_dir, client=s3
    )
    assert main_path.read_bytes() == b"main-bytes"
    assert (dest_dir / "bundle.conv.onnx").read_bytes() == b"conv-bytes"
    assert (dest_dir / "bundle.heads.onnx").read_bytes() == b"heads-bytes"
    assert (dest_dir / "bundle.fc.bin").read_bytes() == b"fc-bytes"


def test_model_downloader_calls_sink_once_per_step(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    bundle_dir = tmp_path / "bundle_src"
    bundle_dir.mkdir()
    (bundle_dir / "latest.onnx").write_bytes(b"main-bytes")
    upload_model_bundle(
        settings=settings,
        onnx_path=bundle_dir / "latest.onnx",
        step=10,
        nn_value_weight=0.5,
        client=s3,
    )

    sink = FakeModelSink()
    downloader = ModelDownloader(
        generator=sink,
        settings=settings,
        local_models_dir=tmp_path / "local_models",
        poll_interval_seconds=10.0,
        client_factory=lambda: s3,
    )
    downloader._poll_once(s3)  # type: ignore[arg-type]
    assert len(sink.calls) == 1
    assert sink.calls[0][1] == 10
    assert sink.calls[0][2] == 0.5

    # Polling again with the same pointer is a no-op.
    downloader._poll_once(s3)  # type: ignore[arg-type]
    assert len(sink.calls) == 1

    # Bumping the step should trigger another sync.
    upload_model_bundle(
        settings=settings,
        onnx_path=bundle_dir / "latest.onnx",
        step=11,
        nn_value_weight=0.6,
        client=s3,
    )
    downloader._poll_once(s3)  # type: ignore[arg-type]
    assert len(sink.calls) == 2
    assert sink.calls[1][1] == 11


def test_offset_table_assigns_sequential_blocks(tmp_path: Path) -> None:
    table = MachineOffsetTable(tmp_path / "offsets.json")
    laptop = table.offset_for("laptop")
    desktop = table.offset_for("desktop")
    laptop_again = table.offset_for("laptop")

    assert laptop == 1 * MACHINE_OFFSET_BLOCK_SIZE  # block 0 is reserved
    assert desktop == 2 * MACHINE_OFFSET_BLOCK_SIZE
    assert laptop_again == laptop  # idempotent
    # First remote game on laptop is a clean 10-digit number.
    assert laptop + 1 == 1_000_000_001


def test_offset_table_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "offsets.json"
    table1 = MachineOffsetTable(path)
    laptop = table1.offset_for("laptop")
    desktop = table1.offset_for("desktop")

    # Reload from disk; existing assignments must be preserved and the next
    # newly-seen machine must continue from the previous max+1.
    table2 = MachineOffsetTable(path)
    assert table2.offset_for("laptop") == laptop
    assert table2.offset_for("desktop") == desktop
    server = table2.offset_for("server")
    assert server == 3 * MACHINE_OFFSET_BLOCK_SIZE


def test_offset_table_rejects_empty_id(tmp_path: Path) -> None:
    table = MachineOffsetTable(tmp_path / "offsets.json")
    with pytest.raises(ValueError):
        table.offset_for("")


def test_game_stats_uploader_writes_json_per_batch(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    source = FakeCompletedGamesSource(
        batches=[
            [
                {
                    "game_number": 1,
                    "stats": {"total_attack": 12.0, "episode_length": 30.0},
                    "completed_time_s": 100.0,
                    "replay": "<should be stripped>",
                },
                {
                    "game_number": 2,
                    "stats": {"total_attack": 7.0, "episode_length": 25.0},
                    "completed_time_s": 102.0,
                },
            ],
            [],  # empty drain — uploader should skip
            [
                {
                    "game_number": 3,
                    "stats": {"total_attack": 20.0, "episode_length": 40.0},
                    "completed_time_s": 110.0,
                }
            ],
        ]
    )
    uploader = GameStatsUploader(
        generator=source,
        settings=settings,
        machine_id="laptop",
        cursor_path=tmp_path / "cursor.json",
        upload_interval_seconds=10.0,
        client_factory=lambda: s3,
    )
    uploader._upload_one_batch(s3)  # type: ignore[arg-type]
    uploader._upload_one_batch(s3)  # type: ignore[arg-type]  # empty
    uploader._upload_one_batch(s3)  # type: ignore[arg-type]

    listing = s3.list_objects_v2(Bucket="bucket", Prefix=settings.game_stats_prefix())
    keys = [obj["Key"] for obj in listing["Contents"]]
    assert keys == [
        f"{settings.game_stats_machine_prefix('laptop')}{0:020d}.json",
        f"{settings.game_stats_machine_prefix('laptop')}{1:020d}.json",
    ]
    body0 = s3.get_object(Bucket="bucket", Key=keys[0])["Body"].read()
    payload0 = json.loads(body0)
    assert [entry["game_number"] for entry in payload0] == [1, 2]
    # `replay` field must be stripped from the upload.
    assert all("replay" not in entry for entry in payload0)


def test_game_stats_round_trip_applies_machine_offset(
    tmp_path: Path, settings: R2Settings
) -> None:
    s3 = InMemoryS3()
    source = FakeCompletedGamesSource(
        batches=[
            [
                {
                    "game_number": 1,
                    "stats": {"total_attack": 5.0, "episode_length": 10.0},
                    "completed_time_s": 10.0,
                },
                {
                    "game_number": 2,
                    "stats": {"total_attack": 6.0, "episode_length": 11.0},
                    "completed_time_s": 11.0,
                },
            ]
        ]
    )
    uploader = GameStatsUploader(
        generator=source,
        settings=settings,
        machine_id="laptop",
        cursor_path=tmp_path / "u_cursor.json",
        upload_interval_seconds=10.0,
        client_factory=lambda: s3,
    )
    uploader._upload_one_batch(s3)  # type: ignore[arg-type]

    sink = FakeGameStatsSink()
    offset_table = MachineOffsetTable(tmp_path / "offsets.json")
    downloader = GameStatsDownloader(
        sink=sink,
        settings=settings,
        cursor_path=tmp_path / "d_cursor.json",
        poll_interval_seconds=10.0,
        offset_table=offset_table,
        client_factory=lambda: s3,
    )
    downloader._poll_once(s3)  # type: ignore[arg-type]

    laptop_offset = offset_table.offset_for("laptop")
    assert len(sink.batches) == 1
    entries, offset, source_machine_id = sink.batches[0]
    assert source_machine_id == "laptop"
    assert offset == laptop_offset
    assert [e["game_number"] for e in entries] == [1, 2]

    # Re-poll: cursor should suppress already-ingested keys.
    downloader._poll_once(s3)  # type: ignore[arg-type]
    assert len(sink.batches) == 1


class _NoSuchKeyOnListClient:
    """Wraps an InMemoryS3 but raises NoSuchKey on list_objects_v2.

    Cloudflare R2 returns NoSuchKey for ListObjectsV2 against a prefix that
    has no objects yet, where AWS S3 returns an empty page. The downloaders
    must treat that as an empty listing instead of crashing.
    """

    def __init__(self, inner: InMemoryS3) -> None:
        self._inner = inner

    def list_objects_v2(self, **_: Any) -> dict[str, Any]:
        raise _MockClientError("NoSuchKey")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_chunk_downloader_tolerates_r2_nosuchkey_on_empty_prefix(
    tmp_path: Path, settings: R2Settings
) -> None:
    client = _NoSuchKeyOnListClient(InMemoryS3())
    sink = FakeReplaySink()
    offset_table = MachineOffsetTable(tmp_path / "offsets.json")
    downloader = ChunkDownloader(
        generator=sink,
        settings=settings,
        cursor_path=tmp_path / "cursor.json",
        poll_interval_seconds=10.0,
        offset_table=offset_table,
        client_factory=lambda: client,  # type: ignore[arg-type]
    )
    downloader._poll_once(client)  # type: ignore[arg-type]
    assert sink.ingested == []


def test_game_stats_downloader_tolerates_r2_nosuchkey_on_empty_prefix(
    tmp_path: Path, settings: R2Settings
) -> None:
    client = _NoSuchKeyOnListClient(InMemoryS3())
    sink = FakeGameStatsSink()
    offset_table = MachineOffsetTable(tmp_path / "offsets.json")
    downloader = GameStatsDownloader(
        sink=sink,
        settings=settings,
        cursor_path=tmp_path / "cursor.json",
        poll_interval_seconds=10.0,
        offset_table=offset_table,
        client_factory=lambda: client,  # type: ignore[arg-type]
    )
    downloader._poll_once(client)  # type: ignore[arg-type]
    assert sink.batches == []


# ---------------------------------------------------------------------------
# discover_active_runs
# ---------------------------------------------------------------------------


def test_discover_active_runs_orders_by_pointer_mtime_desc() -> None:
    """Auto-discovery picks the freshest run; multiple runs are sortable by mtime."""
    from datetime import datetime, timezone

    from tetris_bot.ml.r2_sync import discover_active_runs

    bucket = "bucket"
    prefix = "tetris-mcts"
    # Three runs, each with an incumbent.json. Mtimes assigned manually.
    pointers = {
        "amber-otter-20260503-0930": (
            b'{"step": 100, "nn_value_weight": 0.1, "bundle_prefix": "tetris-mcts/amber-otter-20260503-0930/models/00000000000000000100/"}',
            datetime(2026, 5, 3, 9, 30, tzinfo=timezone.utc),
        ),
        "swift-fox-20260503-0952": (
            b'{"step": 200, "nn_value_weight": 0.2, "bundle_prefix": "tetris-mcts/swift-fox-20260503-0952/models/00000000000000000200/"}',
            datetime(2026, 5, 3, 9, 52, tzinfo=timezone.utc),
        ),
        "ancient-yak-20260502-1200": (
            b'{"step": 50, "nn_value_weight": 0.05, "bundle_prefix": "tetris-mcts/ancient-yak-20260502-1200/models/00000000000000000050/"}',
            datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
        ),
    }

    class _MtimeS3(InMemoryS3):
        def __init__(self) -> None:
            super().__init__()
            self.mtimes: dict[tuple[str, str], "datetime"] = {}

        def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
            with self._lock:
                if (Bucket, Key) not in self._objects:
                    raise _MockClientError("404")
                modified = self.mtimes.get((Bucket, Key))
            return {"LastModified": modified}

    s3 = _MtimeS3()
    for run_id, (body, modified) in pointers.items():
        key = f"{prefix}/{run_id}/models/incumbent.json"
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        s3.mtimes[(bucket, key)] = modified

    runs = discover_active_runs(bucket=bucket, prefix=prefix, client=s3)  # type: ignore[arg-type]
    assert [r.sync_run_id for r in runs] == [
        "swift-fox-20260503-0952",
        "amber-otter-20260503-0930",
        "ancient-yak-20260502-1200",
    ]
    assert runs[0].pointer.step == 200
    assert runs[0].pointer.nn_value_weight == 0.2


def test_discover_active_runs_skips_runs_without_pointer() -> None:
    """Runs that exist as a prefix but have no incumbent.json are dropped."""
    from datetime import datetime, timezone

    from tetris_bot.ml.r2_sync import discover_active_runs

    bucket = "bucket"
    prefix = "tetris-mcts"

    class _MtimeS3(InMemoryS3):
        def __init__(self) -> None:
            super().__init__()
            self.mtimes: dict[tuple[str, str], "datetime"] = {}

        def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
            with self._lock:
                if (Bucket, Key) not in self._objects:
                    raise _MockClientError("404")
                modified = self.mtimes.get((Bucket, Key))
            return {"LastModified": modified}

    s3 = _MtimeS3()
    # Two runs visible as prefixes (via games/ keys), only one has incumbent.json.
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/with-pointer-20260503-0930/games/m/0.json",
        Body=b"{}",
    )
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/no-pointer-20260503-0931/games/m/0.json",
        Body=b"{}",
    )
    pointer_key = f"{prefix}/with-pointer-20260503-0930/models/incumbent.json"
    s3.put_object(
        Bucket=bucket,
        Key=pointer_key,
        Body=b'{"step": 1, "nn_value_weight": 0.1, "bundle_prefix": "tetris-mcts/with-pointer-20260503-0930/models/00000000000000000001/"}',
    )
    s3.mtimes[(bucket, pointer_key)] = datetime(2026, 5, 3, 9, 30, tzinfo=timezone.utc)

    runs = discover_active_runs(bucket=bucket, prefix=prefix, client=s3)  # type: ignore[arg-type]
    assert [r.sync_run_id for r in runs] == ["with-pointer-20260503-0930"]


def test_discover_active_runs_returns_empty_for_fresh_bucket() -> None:
    """No runs at all -> empty list, not an error."""
    from tetris_bot.ml.r2_sync import discover_active_runs

    s3 = InMemoryS3()
    runs = discover_active_runs(bucket="bucket", prefix="tetris-mcts", client=s3)  # type: ignore[arg-type]
    assert runs == []


def test_select_sync_run_id_polls_until_run_appears(monkeypatch) -> None:
    """Generator can join during the trainer's bootstrap phase: it polls
    discovery until at least one run with `models/incumbent.json` appears.
    """
    from datetime import datetime, timezone

    from tetris_bot.ml.config import R2SyncConfig
    from tetris_bot.scripts.run_generator import _select_sync_run_id

    bucket = "bucket"
    prefix = "tetris-mcts"
    pointer_body = (
        b'{"step": 5, "nn_value_weight": 0.2, "bundle_prefix": '
        b'"tetris-mcts/swift-fox-20260503-0952/models/00000000000000000005/"}'
    )

    class _MtimeS3(InMemoryS3):
        def __init__(self) -> None:
            super().__init__()
            self.mtimes: dict[tuple[str, str], "datetime"] = {}

        def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
            with self._lock:
                if (Bucket, Key) not in self._objects:
                    raise _MockClientError("404")
                modified = self.mtimes.get((Bucket, Key))
            return {"LastModified": modified}

    s3 = _MtimeS3()

    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # Simulate the trainer publishing its first incumbent after the
        # second sleep — exercises the poll loop.
        if len(sleep_calls) == 2:
            key = f"{prefix}/swift-fox-20260503-0952/models/incumbent.json"
            s3.put_object(Bucket=bucket, Key=key, Body=pointer_body)
            s3.mtimes[(bucket, key)] = datetime(
                2026, 5, 3, 9, 52, tzinfo=timezone.utc
            )

    monkeypatch.setattr("tetris_bot.scripts.run_generator.time.sleep", _fake_sleep)
    monkeypatch.setattr(
        "tetris_bot.scripts.run_generator.make_s3_client", lambda _settings: s3
    )

    cfg = R2SyncConfig(
        enabled=True,
        bucket=bucket,
        prefix=prefix,
        active_run_freshness_seconds=86400.0,
    )
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://mock.r2")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    chosen = _select_sync_run_id(
        cfg,
        explicit_override=None,
        interactive=False,
        wait_seconds=60.0,
        poll_interval_seconds=0.1,
    )
    assert chosen == "swift-fox-20260503-0952"
    # First two attempts saw no runs; the publish happens during the
    # second sleep, so the third discovery returns the run.
    assert len(sleep_calls) == 2
