from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import string
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_bot.constants import (
    BENCHMARKS_DIR,
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    TRAINING_RUNS_DIR,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import FC_BINARY_MAGIC, export_onnx, export_split_models

logger = structlog.get_logger()
_DEFAULT_SELF_PLAY = load_training_config(DEFAULT_CONFIG_PATH).self_play
_OPTIMIZER_VERSION = 1
_SUPPORTED_BACKENDS = {"tract", "ort"}
_WORKER_SEARCH_MODES = {"adaptive", "grid"}
_BACKEND_STRATEGIES = {"staged", "exhaustive"}
_BENCHMARK_MODEL_PATH = BENCHMARKS_DIR / "models" / "v3_latest.onnx"
_AUTO_MODEL_PATH = BENCHMARKS_DIR / "models" / "optimize_bootstrap.onnx"


@dataclass(frozen=True)
class BuildProfile:
    name: str
    rustflags: str
    lto: str
    codegen_units: str


_BUILD_PROFILES: dict[str, BuildProfile] = {
    "tuned": BuildProfile(
        name="tuned",
        rustflags="-C target-cpu=native",
        lto="thin",
        codegen_units="1",
    ),
    "baseline": BuildProfile(
        name="baseline",
        rustflags="",
        lto="off",
        codegen_units="16",
    ),
}


def _best_effort_sysctl(key: str) -> int | None:
    if platform.system().lower() != "darwin":
        return None
    try:
        completed = subprocess.run(
            ["sysctl", "-n", key],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    raw = completed.stdout.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def detect_cpu_counts() -> tuple[int, int]:
    logical = max(1, os.cpu_count() or 1)
    physical = _best_effort_sysctl("hw.physicalcpu") or logical
    return physical, logical


def default_worker_candidates() -> list[int]:
    physical, logical = detect_cpu_counts()
    preferred = [6, 8, 10, 12, 16, 24, 32, 48, 64]
    values = [value for value in preferred if 4 < value <= logical]
    values.append(physical)
    values.append(logical)
    return sorted(set(value for value in values if value > 4))


def resolve_model_path(model_path_arg: Path | None) -> Path:
    if model_path_arg is not None:
        return model_path_arg
    model_path = default_model_path()
    if model_path is not None:
        return model_path
    return ensure_auto_model_path()


def has_split_model_bundle(model_path: Path) -> bool:
    base = model_path.with_suffix("")
    required = [
        model_path,
        base.with_suffix(".conv.onnx"),
        base.with_suffix(".heads.onnx"),
        base.with_suffix(".fc.bin"),
    ]
    return all(path.exists() for path in required)


def _format_magic_for_error(magic: bytes) -> str:
    if magic and all(chr(byte) in string.printable and byte >= 32 for byte in magic):
        return magic.decode("ascii", errors="replace")
    return magic.hex()


def split_model_bundle_error(model_path: Path) -> str | None:
    base = model_path.with_suffix("")
    conv_path = base.with_suffix(".conv.onnx")
    heads_path = base.with_suffix(".heads.onnx")
    fc_path = base.with_suffix(".fc.bin")
    required = [model_path, conv_path, heads_path, fc_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        return f"Model bundle incomplete for {model_path} (missing {missing_display})"

    magic = fc_path.read_bytes()[: len(FC_BINARY_MAGIC)]
    if magic != FC_BINARY_MAGIC:
        got_magic = _format_magic_for_error(magic)
        expected_magic = FC_BINARY_MAGIC.decode("ascii")
        return (
            f"Incompatible cached board-path binary in {fc_path} "
            f"(expected magic {expected_magic}, got {got_magic})"
        )
    return None


def has_compatible_split_model_bundle(model_path: Path) -> bool:
    return split_model_bundle_error(model_path) is None


def _latest_compatible_training_bundle() -> Path | None:
    checkpoints = sorted(
        TRAINING_RUNS_DIR.glob("v*/checkpoints/latest.onnx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for model_path in checkpoints:
        if has_compatible_split_model_bundle(model_path):
            return model_path
    return None


def default_model_path() -> Path | None:
    benchmark_error = split_model_bundle_error(_BENCHMARK_MODEL_PATH)
    if benchmark_error is None:
        return _BENCHMARK_MODEL_PATH
    if _BENCHMARK_MODEL_PATH.exists():
        logger.warning(
            "Skipping incompatible benchmark optimize bundle",
            model_path=str(_BENCHMARK_MODEL_PATH),
            error=benchmark_error,
        )

    training_model_path = _latest_compatible_training_bundle()
    if training_model_path is not None:
        logger.info(
            "Using latest compatible training bundle for optimization",
            model_path=str(training_model_path),
        )
        return training_model_path
    return None


def ensure_auto_model_path() -> Path:
    if has_compatible_split_model_bundle(_AUTO_MODEL_PATH):
        return _AUTO_MODEL_PATH

    logger.info(
        "Generating bootstrap optimize bundle",
        model_path=str(_AUTO_MODEL_PATH),
    )
    _AUTO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = TetrisNet(**load_training_config(DEFAULT_CONFIG_PATH).network.model_dump())
    model.eval()
    onnx_export_ok = export_onnx(model, _AUTO_MODEL_PATH)
    split_export_ok = export_split_models(model, _AUTO_MODEL_PATH)
    bundle_error = split_model_bundle_error(_AUTO_MODEL_PATH)
    if not onnx_export_ok or not split_export_ok or bundle_error is not None:
        raise RuntimeError(
            "Failed to auto-generate a compatible optimize model bundle at "
            f"{_AUTO_MODEL_PATH}: {bundle_error or 'ONNX export failed'}"
        )
    return _AUTO_MODEL_PATH


def split_model_signature(model_path: Path) -> dict[str, int]:
    base = model_path.with_suffix("")
    files = [
        model_path,
        base.with_suffix(".conv.onnx"),
        base.with_suffix(".heads.onnx"),
        base.with_suffix(".fc.bin"),
    ]
    signature: dict[str, int] = {}
    for path in files:
        stat = path.stat()
        signature[f"{path.name}:size"] = stat.st_size
        signature[f"{path.name}:mtime_ns"] = stat.st_mtime_ns
    return signature


@dataclass
class ScriptArgs:
    """Auto-tune build/backend/worker settings for local game-generation throughput."""

    model_path: Path | None = None
    worker_candidates: list[int] = field(default_factory=default_worker_candidates)
    compile_profiles: list[str] = field(default_factory=lambda: ["tuned", "baseline"])
    backends: list[str] = field(default_factory=lambda: ["tract", "ort"])
    backend_strategy: str = "staged"
    primary_backend: str = "ort"
    worker_search: str = "adaptive"
    max_worker_evals_per_combo: int = 6
    num_games: int = 20
    num_repeats: int = 1
    simulations: int = _DEFAULT_SELF_PLAY.num_simulations
    max_placements: int = _DEFAULT_SELF_PLAY.max_placements
    seed_start: int = 42
    mcts_seed: int = 12345
    c_puct: float = _DEFAULT_SELF_PLAY.c_puct
    force: bool = False
    skip_build: bool = False
    cache_dir: Path = BENCHMARKS_DIR / "profiles" / "optimize_cache"


@dataclass
class BenchmarkResult:
    compile_profile: str
    backend: str
    num_workers: int
    repeat_idx: int
    total_time_sec: float
    games_per_sec: float
    moves_per_sec: float
    avg_moves: float
    avg_attack: float
    max_attack: int
    result_path: str


def machine_profile() -> dict[str, str | int]:
    physical, logical = detect_cpu_counts()
    memsize_bytes = _best_effort_sysctl("hw.memsize")
    profile = {
        "optimizer_version": _OPTIMIZER_VERSION,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_physical": physical,
        "cpu_logical": logical,
        "hostname": socket.gethostname(),
    }
    if memsize_bytes is not None:
        profile["memsize_bytes"] = memsize_bytes
    return profile


def machine_type_fingerprint(profile: dict[str, str | int]) -> str:
    machine_type = {
        "optimizer_version": profile["optimizer_version"],
        "platform_system": profile["platform_system"],
        "platform_release": profile["platform_release"],
        "platform_machine": profile["platform_machine"],
        "cpu_physical": profile["cpu_physical"],
        "cpu_logical": profile["cpu_logical"],
        "memsize_bytes": profile.get("memsize_bytes"),
    }
    raw = json.dumps(machine_type, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def request_hash(
    *,
    profile: dict[str, str | int],
    args: ScriptArgs,
    model_path: Path,
    workers: list[int],
    compile_profiles: list[str],
    backends: list[str],
    backend_strategy: str,
    primary_backend: str,
) -> str:
    request = {
        "optimizer_version": _OPTIMIZER_VERSION,
        "machine_type_fingerprint": machine_type_fingerprint(profile),
        "model_path": str(model_path),
        "model_signature": split_model_signature(model_path),
        "worker_candidates": workers,
        "backend_strategy": backend_strategy,
        "primary_backend": primary_backend,
        "worker_search": args.worker_search,
        "max_worker_evals_per_combo": args.max_worker_evals_per_combo,
        "compile_profiles": compile_profiles,
        "backends": backends,
        "num_games": args.num_games,
        "num_repeats": args.num_repeats,
        "simulations": args.simulations,
        "max_placements": args.max_placements,
        "seed_start": args.seed_start,
        "mcts_seed": args.mcts_seed,
        "c_puct": args.c_puct,
    }
    return hashlib.sha256(
        json.dumps(request, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _venv_python() -> Path:
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def _cargo_env(base_env: dict[str, str]) -> dict[str, str]:
    env = base_env.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"
    # Ensure PyO3 always binds against this repo's venv interpreter, even if the
    # parent shell exported PYO3_PYTHON for a different environment.
    env["PYO3_PYTHON"] = str(_venv_python())
    return env


def build_extension(profile: BuildProfile, *, enable_ort: bool) -> None:
    env = _cargo_env(os.environ.copy())
    env["CARGO_PROFILE_RELEASE_LTO"] = profile.lto
    env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = profile.codegen_units
    env["RUSTFLAGS"] = profile.rustflags
    features = "extension-module,nn-ort" if enable_ort else "extension-module"
    dist_dir = PROJECT_ROOT / "tetris_core" / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)
    build_cmd = [
        str(_venv_python()),
        "-m",
        "maturin",
        "build",
        "--release",
        "--features",
        features,
        "--manifest-path",
        "tetris_core/Cargo.toml",
        "--out",
        str(dist_dir),
    ]
    logger.info(
        "Building extension for profile",
        compile_profile=profile.name,
        ort_enabled=enable_ort,
        rustflags=profile.rustflags,
        lto=profile.lto,
        codegen_units=profile.codegen_units,
    )
    subprocess.run(build_cmd, cwd=PROJECT_ROOT, check=True, env=env)
    wheels = list(dist_dir.glob("tetris_core-*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel found in {dist_dir} after maturin build")
    venv_dir = PROJECT_ROOT / ".venv"
    install_env = {**os.environ, "VIRTUAL_ENV": str(venv_dir)}
    install_cmd = ["uv", "pip", "install", "--no-deps", "--reinstall", str(wheels[0])]
    subprocess.run(install_cmd, cwd=PROJECT_ROOT, check=True, env=install_env)


def run_profile_once(
    *,
    args: ScriptArgs,
    compile_profile: str,
    backend: str,
    workers: int,
    repeat_idx: int,
    model_path: Path,
    result_file: Path,
) -> BenchmarkResult:
    env = os.environ.copy()
    env["TETRIS_NN_BACKEND"] = backend
    seed_start = args.seed_start + repeat_idx * args.num_games
    cmd = [
        str(_venv_python()),
        "tetris_bot/scripts/inspection/profile_games.py",
        "--model_path",
        str(model_path),
        "--num_games",
        str(args.num_games),
        "--simulations",
        str(args.simulations),
        "--num_workers",
        str(workers),
        "--seed_start",
        str(seed_start),
        "--mcts_seed",
        str(args.mcts_seed),
        "--max_placements",
        str(args.max_placements),
        "--c_puct",
        str(args.c_puct),
        "--output",
        str(result_file),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)

    lines = [line for line in result_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No profile output found in {result_file}")
    row = json.loads(lines[-1])
    timing = row["timing"]
    results = row["results"]
    return BenchmarkResult(
        compile_profile=compile_profile,
        backend=backend,
        num_workers=workers,
        repeat_idx=repeat_idx,
        total_time_sec=float(timing["total_time_sec"]),
        games_per_sec=float(timing["games_per_second"]),
        moves_per_sec=float(timing["moves_per_second"]),
        avg_moves=float(results["avg_moves"]),
        avg_attack=float(results["avg_attack"]),
        max_attack=int(results["max_attack"]),
        result_path=str(result_file),
    )


def ensure_valid_args(
    args: ScriptArgs,
) -> tuple[Path, list[int], list[str], list[str], str, str]:
    model_path = resolve_model_path(args.model_path)
    bundle_error = split_model_bundle_error(model_path)
    if bundle_error is not None:
        raise ValueError(bundle_error)
    if args.num_games <= 0:
        raise ValueError(f"num_games must be > 0 (got {args.num_games})")
    if args.num_repeats <= 0:
        raise ValueError(f"num_repeats must be > 0 (got {args.num_repeats})")
    if args.simulations <= 0:
        raise ValueError(f"simulations must be > 0 (got {args.simulations})")
    if args.max_placements <= 0:
        raise ValueError(f"max_placements must be > 0 (got {args.max_placements})")
    if args.max_worker_evals_per_combo <= 0:
        raise ValueError(
            f"max_worker_evals_per_combo must be > 0 (got {args.max_worker_evals_per_combo})"
        )
    if args.worker_search not in _WORKER_SEARCH_MODES:
        raise ValueError(
            f"worker_search must be 'adaptive' or 'grid' (got {args.worker_search})"
        )
    backend_strategy = args.backend_strategy.strip().lower()
    if backend_strategy not in _BACKEND_STRATEGIES:
        raise ValueError(
            "backend_strategy must be 'staged' or 'exhaustive' "
            f"(got {args.backend_strategy})"
        )

    workers = sorted(set(args.worker_candidates))
    if not workers:
        raise ValueError("worker_candidates cannot be empty")
    if any(worker <= 4 for worker in workers):
        raise ValueError(
            f"worker_candidates must all be > 4 (2 and 4 workers are too low; got {workers})"
        )

    compile_profiles = []
    for profile_name in args.compile_profiles:
        if profile_name not in _BUILD_PROFILES:
            raise ValueError(
                f"Unsupported compile profile '{profile_name}'. Supported: {sorted(_BUILD_PROFILES)}"
            )
        compile_profiles.append(profile_name)
    compile_profiles = list(dict.fromkeys(compile_profiles))

    backends = []
    for backend in args.backends:
        backend_lower = backend.strip().lower()
        if backend_lower not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported: ['tract', 'ort']"
            )
        backends.append(backend_lower)
    backends = list(dict.fromkeys(backends))
    primary_backend = args.primary_backend.strip().lower()
    if primary_backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"primary_backend must be one of {sorted(_SUPPORTED_BACKENDS)} "
            f"(got {args.primary_backend})"
        )
    if backend_strategy == "staged" and primary_backend not in backends:
        raise ValueError(
            f"primary_backend '{primary_backend}' is not in backends list {backends}"
        )

    return (
        model_path,
        workers,
        compile_profiles,
        backends,
        backend_strategy,
        primary_backend,
    )


def _cache_path(cache_dir: Path, machine_fingerprint: str) -> Path:
    return cache_dir / f"{machine_fingerprint}.json"


def _env_cache_path(cache_dir: Path, machine_fingerprint: str) -> Path:
    return cache_dir / f"{machine_fingerprint}.env"


def _median_moves_per_sec_for_worker(
    results: list[BenchmarkResult], compile_profile: str, backend: str, worker: int
) -> float:
    matching = [
        row.moves_per_sec
        for row in results
        if row.compile_profile == compile_profile
        and row.backend == backend
        and row.num_workers == worker
    ]
    if not matching:
        raise RuntimeError(
            f"No matching results for compile={compile_profile} backend={backend} worker={worker}"
        )
    matching_sorted = sorted(matching)
    middle = len(matching_sorted) // 2
    if len(matching_sorted) % 2 == 1:
        return matching_sorted[middle]
    return 0.5 * (matching_sorted[middle - 1] + matching_sorted[middle])


def _adaptive_worker_indices(worker_count: int) -> list[int]:
    if worker_count <= 0:
        return []
    candidates = [worker_count // 2, 0, worker_count - 1]
    quarter = worker_count // 4
    if quarter > 0:
        candidates.append(quarter)
        candidates.append(worker_count - 1 - quarter)

    indices: list[int] = []
    for candidate in candidates:
        if 0 <= candidate < worker_count and candidate not in indices:
            indices.append(candidate)
    return indices


def _pick_best_neighbor_worker(
    *,
    workers: list[int],
    compile_profile: str,
    backend: str,
    results: list[BenchmarkResult],
    evaluated_workers: set[int],
) -> int | None:
    if not evaluated_workers:
        return None

    scored = []
    for worker in evaluated_workers:
        score = _median_moves_per_sec_for_worker(
            results, compile_profile, backend, worker
        )
        scored.append((score, worker))
    scored.sort(reverse=True)
    best_worker = scored[0][1]
    best_idx = workers.index(best_worker)

    max_radius = max(best_idx, len(workers) - 1 - best_idx)
    for radius in range(1, max_radius + 1):
        for candidate_idx in (best_idx - radius, best_idx + radius):
            if 0 <= candidate_idx < len(workers):
                candidate = workers[candidate_idx]
                if candidate not in evaluated_workers:
                    return candidate
    return None


def _pick_midpoint_remaining_worker(
    workers: list[int], evaluated_workers: set[int]
) -> int | None:
    remaining = [worker for worker in workers if worker not in evaluated_workers]
    if not remaining:
        return None
    return remaining[len(remaining) // 2]


def load_cached_result(
    *,
    cache_dir: Path,
    machine_fingerprint: str,
    req_hash: str,
) -> dict | None:
    path = _cache_path(cache_dir, machine_fingerprint)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    entries = payload.get("entries", [])
    for entry in entries:
        if entry.get("request_hash") == req_hash:
            return entry
    return None


def write_cache_result(
    *,
    cache_dir: Path,
    machine_fingerprint: str,
    machine: dict[str, str | int],
    request_hash_value: str,
    request_summary: dict,
    results_payload: dict,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, machine_fingerprint)
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = {"machine_profile": machine, "entries": []}

    entries = payload.setdefault("entries", [])
    entries = [
        entry for entry in entries if entry.get("request_hash") != request_hash_value
    ]
    entries.append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_hash": request_hash_value,
            "request": request_summary,
            "results": results_payload,
        }
    )
    payload["machine_profile"] = machine
    payload["entries"] = entries[-20:]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_env_cache(
    *, cache_dir: Path, machine_fingerprint: str, payload: dict
) -> Path:
    best = payload["best"]
    build = best["build_profile"]
    env_lines = [
        "# Auto-generated by make optimize",
        f"TETRIS_OPTIMIZER_VERSION={_OPTIMIZER_VERSION}",
        f"TETRIS_OPT_BUILD_PROFILE={build['name']}",
        f"TETRIS_OPT_RUSTFLAGS={shlex.quote(build['rustflags'])}",
        f"TETRIS_OPT_LTO={build['lto']}",
        f"TETRIS_OPT_CODEGEN_UNITS={build['codegen_units']}",
        f"TETRIS_NN_BACKEND={best['backend']}",
        f"TETRIS_OPT_NUM_WORKERS={best['num_workers']}",
        f"RELEASE_RUSTFLAGS={shlex.quote(build['rustflags'])}",
        f"RELEASE_LTO={build['lto']}",
        f"RELEASE_CODEGEN_UNITS={build['codegen_units']}",
        f"WORKERS_PROFILE={best['num_workers']}",
        "",
    ]
    env_path = _env_cache_path(cache_dir, machine_fingerprint)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(env_lines))
    return env_path


def print_results_table(results: list[BenchmarkResult]) -> None:
    print(
        f"{'Compile':>9}  {'Backend':>7}  {'Workers':>7}  {'Repeat':>6}  {'Moves/s':>9}  {'Games/s':>9}  {'AvgMoves':>9}  {'AvgAtk':>7}"
    )
    print("-" * 80)
    for row in results:
        print(
            f"{row.compile_profile:>9}  "
            f"{row.backend:>7}  "
            f"{row.num_workers:>7}  "
            f"{row.repeat_idx:>6}  "
            f"{row.moves_per_sec:>9.1f}  "
            f"{row.games_per_sec:>9.2f}  "
            f"{row.avg_moves:>9.1f}  "
            f"{row.avg_attack:>7.1f}"
        )


def select_best(results: list[BenchmarkResult]) -> BenchmarkResult:
    return max(
        results,
        key=lambda row: (row.moves_per_sec, row.games_per_sec, -row.total_time_sec),
    )


def _benchmark_combo_workers(
    *,
    args: ScriptArgs,
    results: list[BenchmarkResult],
    workers: list[int],
    compile_profile_name: str,
    backend: str,
    model_path: Path,
    run_output_dir: Path,
    fingerprint: str,
) -> None:
    evaluated_workers: set[int] = set()
    max_evals_for_combo = (
        len(workers)
        if args.worker_search == "grid"
        else min(len(workers), args.max_worker_evals_per_combo)
    )
    pending_workers = (
        workers.copy()
        if args.worker_search == "grid"
        else [workers[index] for index in _adaptive_worker_indices(len(workers))]
    )

    while len(evaluated_workers) < max_evals_for_combo:
        if pending_workers:
            worker = pending_workers.pop(0)
            if worker in evaluated_workers:
                continue
        else:
            worker = _pick_best_neighbor_worker(
                workers=workers,
                compile_profile=compile_profile_name,
                backend=backend,
                results=results,
                evaluated_workers=evaluated_workers,
            )
            if worker is None:
                worker = _pick_midpoint_remaining_worker(workers, evaluated_workers)
            if worker is None:
                break

        logger.info(
            "Benchmarking candidate",
            compile_profile=compile_profile_name,
            backend=backend,
            workers=worker,
            repeats=args.num_repeats,
        )
        for repeat_idx in range(args.num_repeats):
            result_path = run_output_dir / (
                "optimize_runs_"
                f"{fingerprint}_{compile_profile_name}_{backend}_w{worker}.jsonl"
            )
            result = run_profile_once(
                args=args,
                compile_profile=compile_profile_name,
                backend=backend,
                workers=worker,
                repeat_idx=repeat_idx,
                model_path=model_path,
                result_file=result_path,
            )
            results.append(result)
        evaluated_workers.add(worker)

        if args.worker_search == "adaptive":
            neighbor = _pick_best_neighbor_worker(
                workers=workers,
                compile_profile=compile_profile_name,
                backend=backend,
                results=results,
                evaluated_workers=evaluated_workers,
            )
            if (
                neighbor is not None
                and neighbor not in pending_workers
                and neighbor not in evaluated_workers
            ):
                pending_workers.append(neighbor)
            midpoint = _pick_midpoint_remaining_worker(workers, evaluated_workers)
            if (
                midpoint is not None
                and midpoint not in pending_workers
                and midpoint not in evaluated_workers
                and len(pending_workers) < 2
            ):
                pending_workers.append(midpoint)


def _benchmark_single_worker(
    *,
    args: ScriptArgs,
    results: list[BenchmarkResult],
    compile_profile_name: str,
    backend: str,
    worker: int,
    model_path: Path,
    run_output_dir: Path,
    fingerprint: str,
) -> None:
    logger.info(
        "Benchmarking fixed worker candidate",
        compile_profile=compile_profile_name,
        backend=backend,
        workers=worker,
        repeats=args.num_repeats,
    )
    for repeat_idx in range(args.num_repeats):
        result_path = run_output_dir / (
            "optimize_runs_"
            f"{fingerprint}_{compile_profile_name}_{backend}_w{worker}.jsonl"
        )
        result = run_profile_once(
            args=args,
            compile_profile=compile_profile_name,
            backend=backend,
            workers=worker,
            repeat_idx=repeat_idx,
            model_path=model_path,
            result_file=result_path,
        )
        results.append(result)


def main(args: ScriptArgs) -> None:
    (
        model_path,
        workers,
        compile_profiles,
        backends,
        backend_strategy,
        primary_backend,
    ) = ensure_valid_args(args)
    machine = machine_profile()
    fingerprint = machine_type_fingerprint(machine)
    req_hash = request_hash(
        profile=machine,
        args=args,
        model_path=model_path,
        workers=workers,
        compile_profiles=compile_profiles,
        backends=backends,
        backend_strategy=backend_strategy,
        primary_backend=primary_backend,
    )
    request_summary = {
        "optimizer_version": _OPTIMIZER_VERSION,
        "model_path": str(model_path),
        "model_signature": split_model_signature(model_path),
        "worker_candidates": workers,
        "backend_strategy": backend_strategy,
        "primary_backend": primary_backend,
        "worker_search": args.worker_search,
        "max_worker_evals_per_combo": args.max_worker_evals_per_combo,
        "compile_profiles": compile_profiles,
        "backends": backends,
        "num_games": args.num_games,
        "num_repeats": args.num_repeats,
        "simulations": args.simulations,
        "max_placements": args.max_placements,
        "seed_start": args.seed_start,
        "mcts_seed": args.mcts_seed,
        "c_puct": args.c_puct,
    }

    if not args.force:
        cached = load_cached_result(
            cache_dir=args.cache_dir,
            machine_fingerprint=fingerprint,
            req_hash=req_hash,
        )
        if cached is not None:
            logger.info(
                "Using cached optimization result",
                cache_dir=str(args.cache_dir),
                machine_fingerprint=fingerprint,
                request_hash=req_hash,
            )
            cached_results = cached["results"]
            env_path = write_env_cache(
                cache_dir=args.cache_dir,
                machine_fingerprint=fingerprint,
                payload=cached_results,
            )
            best = cached_results["best"]
            print(
                "Cache hit. Best config:\n"
                f"  build={best['build_profile']['name']} backend={best['backend']} workers={best['num_workers']}\n"
                f"  moves_per_sec={best['moves_per_sec']:.1f} games_per_sec={best['games_per_sec']:.2f}\n"
                f"  env={env_path}"
            )
            return

    logger.info(
        "Starting machine optimization",
        machine_fingerprint=fingerprint,
        model_path=str(model_path),
        worker_candidates=workers,
        compile_profiles=compile_profiles,
        backends=backends,
        backend_strategy=backend_strategy,
        primary_backend=primary_backend,
        worker_search=args.worker_search,
        max_worker_evals_per_combo=args.max_worker_evals_per_combo,
        num_games=args.num_games,
        num_repeats=args.num_repeats,
        simulations=args.simulations,
    )

    results: list[BenchmarkResult] = []
    run_output_dir = BENCHMARKS_DIR / "profiles"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    needs_ort_build = "ort" in backends
    if backend_strategy == "exhaustive":
        for compile_profile_name in compile_profiles:
            build_profile = _BUILD_PROFILES[compile_profile_name]
            if not args.skip_build:
                build_extension(build_profile, enable_ort=needs_ort_build)
            for backend in backends:
                _benchmark_combo_workers(
                    args=args,
                    results=results,
                    workers=workers,
                    compile_profile_name=compile_profile_name,
                    backend=backend,
                    model_path=model_path,
                    run_output_dir=run_output_dir,
                    fingerprint=fingerprint,
                )
    else:
        for compile_profile_name in compile_profiles:
            build_profile = _BUILD_PROFILES[compile_profile_name]
            if not args.skip_build:
                build_extension(build_profile, enable_ort=needs_ort_build)
            _benchmark_combo_workers(
                args=args,
                results=results,
                workers=workers,
                compile_profile_name=compile_profile_name,
                backend=primary_backend,
                model_path=model_path,
                run_output_dir=run_output_dir,
                fingerprint=fingerprint,
            )

        primary_results = [row for row in results if row.backend == primary_backend]
        if not primary_results:
            raise RuntimeError(
                f"No primary backend results were executed for backend={primary_backend}"
            )
        best_primary = select_best(primary_results)
        best_worker = best_primary.num_workers
        best_compile_profile = best_primary.compile_profile
        best_build_profile = _BUILD_PROFILES[best_compile_profile]

        logger.info(
            "Primary backend search complete; comparing backends at chosen worker count",
            primary_backend=primary_backend,
            best_compile_profile=best_compile_profile,
            best_workers=best_worker,
            compare_backends=backends,
        )
        if not args.skip_build:
            build_extension(best_build_profile, enable_ort=needs_ort_build)

        for backend in backends:
            if backend == primary_backend:
                continue
            _benchmark_single_worker(
                args=args,
                results=results,
                compile_profile_name=best_compile_profile,
                backend=backend,
                worker=best_worker,
                model_path=model_path,
                run_output_dir=run_output_dir,
                fingerprint=fingerprint,
            )

    if not results:
        raise RuntimeError("No optimization runs were executed")

    best = select_best(results)
    best_build = _BUILD_PROFILES[best.compile_profile]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_version": _OPTIMIZER_VERSION,
        "machine_profile": machine,
        "machine_fingerprint": fingerprint,
        "request_hash": req_hash,
        "model_path": str(model_path),
        "request": request_summary,
        "runs": [asdict(row) for row in results],
        "best": {
            "compile_profile": best.compile_profile,
            "backend": best.backend,
            "num_workers": best.num_workers,
            "moves_per_sec": best.moves_per_sec,
            "games_per_sec": best.games_per_sec,
            "avg_moves": best.avg_moves,
            "avg_attack": best.avg_attack,
            "result_path": best.result_path,
            "build_profile": asdict(best_build),
        },
    }

    cache_path = write_cache_result(
        cache_dir=args.cache_dir,
        machine_fingerprint=fingerprint,
        machine=machine,
        request_hash_value=req_hash,
        request_summary=request_summary,
        results_payload=payload,
    )
    env_path = write_env_cache(
        cache_dir=args.cache_dir,
        machine_fingerprint=fingerprint,
        payload=payload,
    )

    print("\n=== Optimization Results ===")
    print_results_table(results)
    print("\nBest config:")
    print(
        f"  build={best.compile_profile} backend={best.backend} workers={best.num_workers}"
    )
    print(
        f"  moves_per_sec={best.moves_per_sec:.1f} games_per_sec={best.games_per_sec:.2f}"
    )
    print(f"  cache={cache_path}")
    print(f"  env={env_path}")


if __name__ == "__main__":
    main(parse(ScriptArgs))
