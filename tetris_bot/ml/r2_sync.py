"""Cloudflare R2 (S3-compatible) sync for multi-machine training.

Architecture:
- One trainer machine (Vast.ai/GPU) runs the trainer, exports ONNX bundles, and
  ingests replay chunks + per-game stats produced by remote generators.
- Zero or more generator machines (e.g., a personal laptop) run self-play
  workers, upload completed games as small NPZ chunks plus per-game stats
  JSON, and pull new ONNX bundles when promoted.

Sync layer (R2 keys):
- `<prefix>/<run_id>/replay/<machine_id>/<slice_start:020d>.npz` — replay chunks
- `<prefix>/<run_id>/games/<machine_id>/<seq:020d>.json` — per-game stats
- `<prefix>/<run_id>/models/<step:020d>/bundle.onnx` (+ `.conv.onnx`, `.heads.onnx`, `.fc.bin`)
- `<prefix>/<run_id>/models/incumbent.json` — atomic pointer to current model

Background threads keep the network off the trainer/generator hot paths:
- ChunkUploader: drains `replay_buffer_delta` -> NPZ -> R2 PUT
- GameStatsUploader: drains `drain_completed_games` -> JSON -> R2 PUT
- ChunkDownloader: lists new keys -> downloads -> `ingest_examples_from_npz`
- GameStatsDownloader: lists new keys -> downloads -> sink callback
- ModelDownloader: polls pointer -> downloads bundle -> `sync_model_directly`

Game-number namespacing: each remote machine_id is assigned a 1B-sized
block on first sight by the trainer-side `MachineOffsetTable`; the
mapping persists in the run dir so the same machine keeps the same
block across resumes. Trainer's local games occupy block 0 (game numbers
1, 2, 3, …). The first remote machine occupies block 1 (1_000_000_001,
1_000_000_002, …), etc., so per-game W&B X-axes stay readable.

Auth: credentials come from `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` env
vars. The R2SyncConfig only carries non-secret fields.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import structlog

from tetris_bot.ml.config import R2SyncConfig

logger = structlog.get_logger(__name__)


# ---- Per-machine game-number offset --------------------------------------


# 1 billion game-numbers per machine block. Block 0 is reserved for the
# trainer's own local games (its `games_generated` AtomicU64 grows naturally
# inside [0, BLOCK_SIZE) and cannot realistically reach 1B in a run). Each
# new remote machine_id gets the next sequential block, so game numbers
# stay readable in W&B (laptop's first game is 1_000_000_001, not a 17-digit
# blake2 hash).
MACHINE_OFFSET_BLOCK_SIZE = 1_000_000_000


class MachineOffsetTable:
    """Persistent first-sight allocator that maps `machine_id` to a 1B block.

    Block 0 is reserved for the trainer; remote machines are assigned blocks
    1, 2, 3, … in the order they first appear. The mapping is persisted as
    JSON so resumed runs reuse the same numbering for the same machines, and
    so cursor-tracked re-ingests of old chunks land back in the same block.

    Thread-safe: looked up from the chunk and game-stats download threads.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        raw = _load_json_cursor(path)
        blocks = raw.get("blocks", {})
        if not isinstance(blocks, dict):
            blocks = {}
        self._blocks: dict[str, int] = {
            str(k): int(v) for k, v in blocks.items() if int(v) > 0
        }
        self._next_block = (
            max(self._blocks.values()) + 1 if self._blocks else 1
        )

    def offset_for(self, machine_id: str) -> int:
        if not machine_id:
            raise ValueError("machine_id must be a non-empty string")
        with self._lock:
            block = self._blocks.get(machine_id)
            if block is None:
                block = self._next_block
                self._next_block += 1
                self._blocks[machine_id] = block
                _save_json_cursor(self._path, {"blocks": self._blocks})
        return block * MACHINE_OFFSET_BLOCK_SIZE

    def known_machines(self) -> dict[str, int]:
        with self._lock:
            return dict(self._blocks)


# ---- Settings -------------------------------------------------------------


@dataclass(frozen=True)
class R2Settings:
    """Materialized R2 connection settings (config + env-sourced credentials)."""

    endpoint_url: str
    bucket: str
    prefix: str
    sync_run_id: str
    access_key_id: str
    secret_access_key: str
    request_timeout_seconds: float

    @classmethod
    def from_config(
        cls, cfg: R2SyncConfig, default_run_id: str | None = None
    ) -> "R2Settings":
        if cfg.endpoint_url is None:
            raise ValueError("r2_sync.endpoint_url is required when role != 'off'")
        if cfg.bucket is None:
            raise ValueError("r2_sync.bucket is required when role != 'off'")
        sync_run_id = cfg.sync_run_id or default_run_id
        if sync_run_id is None:
            raise ValueError(
                "r2_sync.sync_run_id is required (or pass a default_run_id)"
            )
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        if not access_key_id or not secret_access_key:
            raise ValueError(
                "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY env vars must be set"
            )
        return cls(
            endpoint_url=cfg.endpoint_url,
            bucket=cfg.bucket,
            prefix=cfg.prefix.strip("/"),
            sync_run_id=sync_run_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            request_timeout_seconds=cfg.request_timeout_seconds,
        )

    def replay_prefix(self) -> str:
        return f"{self.prefix}/{self.sync_run_id}/replay/"

    def replay_machine_prefix(self, machine_id: str) -> str:
        return f"{self.replay_prefix()}{machine_id}/"

    def replay_chunk_key(self, machine_id: str, slice_start: int) -> str:
        return f"{self.replay_machine_prefix(machine_id)}{slice_start:020d}.npz"

    def game_stats_prefix(self) -> str:
        return f"{self.prefix}/{self.sync_run_id}/games/"

    def game_stats_machine_prefix(self, machine_id: str) -> str:
        return f"{self.game_stats_prefix()}{machine_id}/"

    def game_stats_chunk_key(self, machine_id: str, seq: int) -> str:
        return f"{self.game_stats_machine_prefix(machine_id)}{seq:020d}.json"

    def model_prefix(self) -> str:
        return f"{self.prefix}/{self.sync_run_id}/models/"

    def model_bundle_prefix(self, step: int) -> str:
        return f"{self.model_prefix()}{step:020d}/"

    def model_pointer_key(self) -> str:
        return f"{self.model_prefix()}incumbent.json"


# ---- boto3 client wrapper -------------------------------------------------


class S3LikeClient(Protocol):
    """Subset of the boto3 S3 client surface we use (also matches mocks)."""

    def put_object(self, **kwargs: Any) -> Any: ...
    def get_object(self, **kwargs: Any) -> Any: ...
    def head_object(self, **kwargs: Any) -> Any: ...
    def list_objects_v2(self, **kwargs: Any) -> Any: ...
    def upload_file(
        self, Filename: str, Bucket: str, Key: str, **kwargs: Any
    ) -> Any: ...
    def download_file(
        self, Bucket: str, Key: str, Filename: str, **kwargs: Any
    ) -> Any: ...


def make_s3_client(settings: R2Settings) -> S3LikeClient:
    """Create a boto3 S3 client pointed at R2.

    boto3 clients are only thread-safe across read operations; create one per
    worker thread that performs writes.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(  # type: ignore[return-value]
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        region_name="auto",
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=settings.request_timeout_seconds,
            read_timeout=settings.request_timeout_seconds,
        ),
    )


# ---- Cursor persistence --------------------------------------------------


def _load_json_cursor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("r2_sync.cursor_corrupt_resetting", path=str(path))
        return {}


def _save_json_cursor(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


# ---- Chunk uploader (generator side) -------------------------------------


class ReplayDeltaSource(Protocol):
    """Adapter for the bits of GameGenerator we need."""

    def buffer_size(self) -> int: ...
    def dump_replay_delta_to_npz(
        self, filepath: str, from_index: int, max_examples: int
    ) -> tuple[int, int, int, int] | None: ...


class ChunkUploader:
    """Background thread that uploads replay deltas as NPZ chunks.

    The cursor file persists `next_from_index`, the smallest logical index we
    have not yet uploaded, so resuming the generator does not re-upload work.
    """

    def __init__(
        self,
        *,
        generator: ReplayDeltaSource,
        settings: R2Settings,
        machine_id: str,
        cursor_path: Path,
        chunk_max_examples: int,
        upload_interval_seconds: float,
        client_factory: Callable[[], S3LikeClient] | None = None,
    ):
        self._generator = generator
        self._settings = settings
        self._machine_id = machine_id
        self._cursor_path = cursor_path
        self._chunk_max_examples = chunk_max_examples
        self._upload_interval = upload_interval_seconds
        self._client_factory = client_factory or (lambda: make_s3_client(settings))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_from_index = self._load_cursor()

    def _load_cursor(self) -> int:
        data = _load_json_cursor(self._cursor_path)
        return int(data.get("next_from_index", 0))

    def _save_cursor(self) -> None:
        _save_json_cursor(
            self._cursor_path, {"next_from_index": self._next_from_index}
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="r2-chunk-uploader", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        client = self._client_factory()
        while not self._stop.is_set():
            try:
                self._upload_one_chunk(client)
            except Exception:
                logger.exception("r2_sync.chunk_upload_failed")
            self._stop.wait(self._upload_interval)

    def _upload_one_chunk(self, client: S3LikeClient) -> None:
        if self._generator.buffer_size() == 0:
            return
        with tempfile.TemporaryDirectory(prefix="r2-chunk-") as tmp_dir:
            tmp_path = Path(tmp_dir) / "chunk.npz"
            result = self._generator.dump_replay_delta_to_npz(
                str(tmp_path), self._next_from_index, self._chunk_max_examples
            )
            if result is None:
                return
            window_start, window_end, slice_start, count = result
            if count == 0:
                advanced = max(self._next_from_index, window_start)
                if advanced != self._next_from_index:
                    self._next_from_index = advanced
                    self._save_cursor()
                return
            key = self._settings.replay_chunk_key(self._machine_id, slice_start)
            client.upload_file(str(tmp_path), self._settings.bucket, key)
            new_from_index = slice_start + count
            self._next_from_index = new_from_index
            self._save_cursor()
            logger.info(
                "r2_sync.chunk_uploaded",
                key=key,
                count=count,
                slice_start=slice_start,
                window_start=window_start,
                window_end=window_end,
            )


# ---- Chunk downloader (trainer side) -------------------------------------


class ReplayIngestSink(Protocol):
    def ingest_examples_from_npz(
        self, filepath: str, game_number_offset: int = 0
    ) -> int: ...


class ChunkDownloader:
    """Background thread that lists and ingests replay chunks from R2.

    Per-machine cursor records the alphabetically-greatest key already
    ingested. Listing uses `StartAfter` so we never re-pull old chunks.
    """

    def __init__(
        self,
        *,
        generator: ReplayIngestSink,
        settings: R2Settings,
        cursor_path: Path,
        poll_interval_seconds: float,
        offset_table: MachineOffsetTable,
        client_factory: Callable[[], S3LikeClient] | None = None,
    ):
        self._generator = generator
        self._settings = settings
        self._cursor_path = cursor_path
        self._poll_interval = poll_interval_seconds
        self._offset_table = offset_table
        self._client_factory = client_factory or (lambda: make_s3_client(settings))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor: dict[str, str] = self._load_cursor()

    def _load_cursor(self) -> dict[str, str]:
        data = _load_json_cursor(self._cursor_path)
        per_machine = data.get("per_machine", {})
        if not isinstance(per_machine, dict):
            return {}
        return {str(k): str(v) for k, v in per_machine.items()}

    def _save_cursor(self) -> None:
        _save_json_cursor(self._cursor_path, {"per_machine": self._cursor})

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="r2-chunk-downloader", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        client = self._client_factory()
        while not self._stop.is_set():
            try:
                self._poll_once(client)
            except Exception:
                logger.exception("r2_sync.chunk_download_failed")
            self._stop.wait(self._poll_interval)

    def _poll_once(self, client: S3LikeClient) -> None:
        machine_ids = self._discover_machine_ids(client)
        for machine_id in machine_ids:
            self._ingest_machine_new_chunks(client, machine_id)

    def _discover_machine_ids(self, client: S3LikeClient) -> list[str]:
        prefix = self._settings.replay_prefix()
        machine_ids: set[str] = set()
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._settings.bucket,
                "Prefix": prefix,
                "Delimiter": "/",
            }
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            for entry in response.get("CommonPrefixes", []) or []:
                sub_prefix = entry.get("Prefix")
                if not sub_prefix:
                    continue
                rel = sub_prefix[len(prefix) :].rstrip("/")
                if rel:
                    machine_ids.add(rel)
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
        return sorted(machine_ids)

    def _ingest_machine_new_chunks(
        self, client: S3LikeClient, machine_id: str
    ) -> None:
        prefix = self._settings.replay_machine_prefix(machine_id)
        start_after = self._cursor.get(machine_id)
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._settings.bucket,
                "Prefix": prefix,
            }
            if start_after is not None:
                kwargs["StartAfter"] = start_after
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []) or []:
                if self._stop.is_set():
                    return
                key = obj.get("Key")
                if not key or not key.endswith(".npz"):
                    continue
                self._download_and_ingest(client, key, machine_id)
                start_after = key
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")

    def _download_and_ingest(
        self, client: S3LikeClient, key: str, machine_id: str
    ) -> None:
        offset = self._offset_table.offset_for(machine_id)
        with tempfile.TemporaryDirectory(prefix="r2-ingest-") as tmp_dir:
            tmp_path = Path(tmp_dir) / "chunk.npz"
            client.download_file(self._settings.bucket, key, str(tmp_path))
            count = self._generator.ingest_examples_from_npz(
                str(tmp_path), offset
            )
        self._cursor[machine_id] = key
        self._save_cursor()
        logger.info(
            "r2_sync.chunk_ingested", key=key, machine_id=machine_id, count=count
        )


# ---- Game-stats uploader (generator side) -------------------------------


class CompletedGamesSource(Protocol):
    """Adapter for the bits of GameGenerator we need on the upload side."""

    def drain_completed_games(self) -> list[dict[str, Any]]: ...


def _strip_replay_field(entry: dict[str, Any]) -> dict[str, Any]:
    """Drop the non-JSON-serializable `replay` field; everything else is plain.

    Replay frames are a Rust PyClass that doesn't round-trip through JSON;
    skipping them here means remote games appear in W&B per-game logs without
    a GIF. Per-game numeric stats survive round-trip cleanly.
    """
    return {k: v for k, v in entry.items() if k != "replay"}


class GameStatsUploader:
    """Background thread that uploads completed-game stats as JSON.

    Drains `drain_completed_games()` periodically, strips `replay` (not JSON
    friendly), and PUTs each batch under
    `<prefix>/<run_id>/games/<machine_id>/<seq:020d>.json`. The batch is also
    skipped when the drain returns empty.
    """

    def __init__(
        self,
        *,
        generator: CompletedGamesSource,
        settings: R2Settings,
        machine_id: str,
        cursor_path: Path,
        upload_interval_seconds: float,
        client_factory: Callable[[], S3LikeClient] | None = None,
    ):
        self._generator = generator
        self._settings = settings
        self._machine_id = machine_id
        self._cursor_path = cursor_path
        self._upload_interval = upload_interval_seconds
        self._client_factory = client_factory or (lambda: make_s3_client(settings))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_seq = self._load_cursor()

    def _load_cursor(self) -> int:
        data = _load_json_cursor(self._cursor_path)
        return int(data.get("next_seq", 0))

    def _save_cursor(self) -> None:
        _save_json_cursor(self._cursor_path, {"next_seq": self._next_seq})

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="r2-game-stats-uploader", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        client = self._client_factory()
        while not self._stop.is_set():
            try:
                self._upload_one_batch(client)
            except Exception:
                logger.exception("r2_sync.game_stats_upload_failed")
            self._stop.wait(self._upload_interval)

    def _upload_one_batch(self, client: S3LikeClient) -> None:
        drained = self._generator.drain_completed_games()
        if not drained:
            return
        payload = [_strip_replay_field(entry) for entry in drained]
        body = json.dumps(payload, sort_keys=False).encode("utf-8")
        key = self._settings.game_stats_chunk_key(self._machine_id, self._next_seq)
        client.put_object(
            Bucket=self._settings.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        self._next_seq += 1
        self._save_cursor()
        logger.info(
            "r2_sync.game_stats_uploaded",
            key=key,
            count=len(payload),
            seq=self._next_seq - 1,
        )


# ---- Game-stats downloader (trainer side) -------------------------------


class GameStatsSink(Protocol):
    """Trainer-side handler for batches of remote completed-game payloads.

    Implementations should apply `game_number_offset` to each entry's
    `game_number` and stash the entries on a thread-safe queue that the
    trainer's main loop drains alongside local games.
    """

    def push_remote_completed_games(
        self, entries: list[dict[str, Any]], game_number_offset: int
    ) -> None: ...


class GameStatsDownloader:
    """Background thread that fetches per-game stats JSON from R2.

    Per-machine cursor records the alphabetically-greatest key already pushed
    to the sink. Listing uses `StartAfter` so we never re-pull old batches.
    """

    def __init__(
        self,
        *,
        sink: GameStatsSink,
        settings: R2Settings,
        cursor_path: Path,
        poll_interval_seconds: float,
        offset_table: MachineOffsetTable,
        client_factory: Callable[[], S3LikeClient] | None = None,
    ):
        self._sink = sink
        self._settings = settings
        self._cursor_path = cursor_path
        self._poll_interval = poll_interval_seconds
        self._offset_table = offset_table
        self._client_factory = client_factory or (lambda: make_s3_client(settings))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor: dict[str, str] = self._load_cursor()

    def _load_cursor(self) -> dict[str, str]:
        data = _load_json_cursor(self._cursor_path)
        per_machine = data.get("per_machine", {})
        if not isinstance(per_machine, dict):
            return {}
        return {str(k): str(v) for k, v in per_machine.items()}

    def _save_cursor(self) -> None:
        _save_json_cursor(self._cursor_path, {"per_machine": self._cursor})

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="r2-game-stats-downloader", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        client = self._client_factory()
        while not self._stop.is_set():
            try:
                self._poll_once(client)
            except Exception:
                logger.exception("r2_sync.game_stats_download_failed")
            self._stop.wait(self._poll_interval)

    def _poll_once(self, client: S3LikeClient) -> None:
        for machine_id in self._discover_machine_ids(client):
            self._pull_machine_new_batches(client, machine_id)

    def _discover_machine_ids(self, client: S3LikeClient) -> list[str]:
        prefix = self._settings.game_stats_prefix()
        machine_ids: set[str] = set()
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._settings.bucket,
                "Prefix": prefix,
                "Delimiter": "/",
            }
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            for entry in response.get("CommonPrefixes", []) or []:
                sub_prefix = entry.get("Prefix")
                if not sub_prefix:
                    continue
                rel = sub_prefix[len(prefix) :].rstrip("/")
                if rel:
                    machine_ids.add(rel)
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
        return sorted(machine_ids)

    def _pull_machine_new_batches(
        self, client: S3LikeClient, machine_id: str
    ) -> None:
        offset = self._offset_table.offset_for(machine_id)
        prefix = self._settings.game_stats_machine_prefix(machine_id)
        start_after = self._cursor.get(machine_id)
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._settings.bucket,
                "Prefix": prefix,
            }
            if start_after is not None:
                kwargs["StartAfter"] = start_after
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []) or []:
                if self._stop.is_set():
                    return
                key = obj.get("Key")
                if not key or not key.endswith(".json"):
                    continue
                self._download_and_push(client, key, machine_id, offset)
                start_after = key
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")

    def _download_and_push(
        self,
        client: S3LikeClient,
        key: str,
        machine_id: str,
        game_number_offset: int,
    ) -> None:
        response = client.get_object(Bucket=self._settings.bucket, Key=key)
        body = response["Body"].read()
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            logger.exception("r2_sync.game_stats_payload_parse_failed", key=key)
            return
        if not isinstance(payload, list):
            logger.warning(
                "r2_sync.game_stats_payload_not_list",
                key=key,
                payload_type=type(payload).__name__,
            )
            return
        self._sink.push_remote_completed_games(payload, game_number_offset)
        self._cursor[machine_id] = key
        self._save_cursor()
        logger.info(
            "r2_sync.game_stats_ingested",
            key=key,
            machine_id=machine_id,
            count=len(payload),
        )


# ---- Model bundle push/pull ---------------------------------------------


@dataclass(frozen=True)
class ModelPointer:
    step: int
    nn_value_weight: float
    bundle_prefix: str

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "step": self.step,
                "nn_value_weight": self.nn_value_weight,
                "bundle_prefix": self.bundle_prefix,
            },
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "ModelPointer":
        data = json.loads(raw.decode("utf-8"))
        return cls(
            step=int(data["step"]),
            nn_value_weight=float(data["nn_value_weight"]),
            bundle_prefix=str(data["bundle_prefix"]),
        )


def _bundle_member_paths(onnx_path: Path) -> list[Path]:
    """Return [main, conv, heads, fc] paths if they exist; main is required."""
    base = onnx_path.with_suffix("")
    members = [
        onnx_path,
        base.with_suffix(".conv.onnx"),
        base.with_suffix(".heads.onnx"),
        base.with_suffix(".fc.bin"),
    ]
    if not members[0].exists():
        raise FileNotFoundError(f"Main ONNX not found: {onnx_path}")
    return [p for p in members if p.exists()]


def upload_model_bundle(
    *,
    settings: R2Settings,
    onnx_path: Path,
    step: int,
    nn_value_weight: float,
    client: S3LikeClient | None = None,
) -> ModelPointer:
    """Upload a model bundle and atomically swap the incumbent pointer.

    Caller must ensure the bundle on disk is complete (all 4 files written)
    before calling this. Pointer write happens last so workers never see a
    partial bundle.
    """
    own_client = client is None
    if own_client:
        client = make_s3_client(settings)
    assert client is not None
    bundle_prefix = settings.model_bundle_prefix(step)
    main_key_basename = "bundle.onnx"
    suffix_to_basename = {
        "": main_key_basename,
        ".conv.onnx": "bundle.conv.onnx",
        ".heads.onnx": "bundle.heads.onnx",
        ".fc.bin": "bundle.fc.bin",
    }
    base = onnx_path.with_suffix("")
    member_map: dict[str, Path] = {}
    member_map[main_key_basename] = onnx_path
    for suffix in (".conv.onnx", ".heads.onnx", ".fc.bin"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            member_map[suffix_to_basename[suffix]] = candidate
    for basename, path in member_map.items():
        client.upload_file(str(path), settings.bucket, bundle_prefix + basename)
    pointer = ModelPointer(
        step=step,
        nn_value_weight=nn_value_weight,
        bundle_prefix=bundle_prefix,
    )
    client.put_object(
        Bucket=settings.bucket,
        Key=settings.model_pointer_key(),
        Body=pointer.to_json(),
        ContentType="application/json",
    )
    logger.info(
        "r2_sync.model_uploaded",
        step=step,
        nn_value_weight=nn_value_weight,
        bundle_prefix=bundle_prefix,
        members=list(member_map.keys()),
    )
    return pointer


def download_model_bundle(
    *,
    settings: R2Settings,
    pointer: ModelPointer,
    dest_dir: Path,
    client: S3LikeClient | None = None,
) -> Path:
    """Download a bundle to `dest_dir`. Returns the local main ONNX path."""
    own_client = client is None
    if own_client:
        client = make_s3_client(settings)
    assert client is not None
    dest_dir.mkdir(parents=True, exist_ok=True)
    main_key = pointer.bundle_prefix + "bundle.onnx"
    main_path = dest_dir / "bundle.onnx"
    client.download_file(settings.bucket, main_key, str(main_path))
    for suffix in (".conv.onnx", ".heads.onnx", ".fc.bin"):
        member_basename = f"bundle{suffix}"
        member_key = pointer.bundle_prefix + member_basename
        member_path = dest_dir / member_basename
        try:
            client.download_file(settings.bucket, member_key, str(member_path))
        except Exception as e:
            from botocore.exceptions import ClientError

            if isinstance(e, ClientError) and e.response.get("Error", {}).get(
                "Code"
            ) in ("404", "NoSuchKey"):
                # Optional split-model files; main ONNX is sufficient if absent.
                continue
            raise
    return main_path


def fetch_model_pointer(
    settings: R2Settings, client: S3LikeClient | None = None
) -> ModelPointer | None:
    own_client = client is None
    if own_client:
        client = make_s3_client(settings)
    assert client is not None
    from botocore.exceptions import ClientError

    try:
        response = client.get_object(
            Bucket=settings.bucket, Key=settings.model_pointer_key()
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            return None
        raise
    body = response["Body"].read()
    return ModelPointer.from_json(body)


# ---- Model downloader (generator side) -----------------------------------


class ModelSyncSink(Protocol):
    def sync_model_directly(
        self, model_path: str, model_step: int, nn_value_weight: float
    ) -> bool: ...


class ModelDownloader:
    """Background thread that polls the incumbent pointer and applies new models.

    The generator must be configured with `candidate_gating_enabled=False`
    because it accepts the trainer's promotion decisions verbatim.
    """

    def __init__(
        self,
        *,
        generator: ModelSyncSink,
        settings: R2Settings,
        local_models_dir: Path,
        poll_interval_seconds: float,
        initial_model_step: int = 0,
        client_factory: Callable[[], S3LikeClient] | None = None,
    ):
        self._generator = generator
        self._settings = settings
        self._local_models_dir = local_models_dir
        self._poll_interval = poll_interval_seconds
        self._client_factory = client_factory or (lambda: make_s3_client(settings))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_synced_step = initial_model_step

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="r2-model-downloader", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        client = self._client_factory()
        while not self._stop.is_set():
            try:
                self._poll_once(client)
            except Exception:
                logger.exception("r2_sync.model_download_failed")
            self._stop.wait(self._poll_interval)

    def _poll_once(self, client: S3LikeClient) -> None:
        pointer = fetch_model_pointer(self._settings, client=client)
        if pointer is None:
            return
        if pointer.step <= self._last_synced_step:
            return
        bundle_dir = self._local_models_dir / f"step_{pointer.step:020d}"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        try:
            main_path = download_model_bundle(
                settings=self._settings,
                pointer=pointer,
                dest_dir=bundle_dir,
                client=client,
            )
        except Exception:
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir, ignore_errors=True)
            raise
        applied = self._generator.sync_model_directly(
            str(main_path), pointer.step, pointer.nn_value_weight
        )
        if applied:
            self._last_synced_step = pointer.step
            logger.info(
                "r2_sync.model_synced",
                step=pointer.step,
                nn_value_weight=pointer.nn_value_weight,
                local_path=str(main_path),
            )
        else:
            logger.info("r2_sync.model_sync_ignored_stale", step=pointer.step)
