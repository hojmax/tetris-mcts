from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import time
from typing import cast

import structlog
import torch
import wandb

from tetris_mcts.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FRAME_DURATION_MS,
    INCUMBENT_ONNX_FILENAME,
    LATEST_ONNX_FILENAME,
    MODEL_CANDIDATES_DIRNAME,
    PARALLEL_ONNX_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_mcts.config import TrainingConfig
from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.weights import (
    WeightManager,
    export_onnx,
    export_split_models,
    split_model_paths,
)
from tetris_mcts.ml.loss import RunningLossBalancer, compute_loss, compute_metrics
from tetris_mcts.ml.visualization import create_trajectory_gif, render_replay
from tetris_mcts.ml.artifacts import (
    assert_rust_inference_artifacts,
    copy_model_artifact_bundle,
)
from tetris_mcts.ml.replay_buffer import TrainingBatch, CircularReplayMirror
from tetris_mcts.ml.game_metrics import (
    compute_batch_feature_metrics,
    summarize_completed_games,
)

from tetris_core import MCTSConfig, GameGenerator

logger = structlog.get_logger()


def roll_interval_deadline(deadline_s: float, interval_s: float, now_s: float) -> float:
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0 (got {interval_s})")
    if now_s < deadline_s:
        return deadline_s
    elapsed_intervals = int((now_s - deadline_s) // interval_s) + 1
    return deadline_s + elapsed_intervals * interval_s


class Trainer:
    """Main training class."""

    def __init__(
        self,
        config: TrainingConfig,
        model: TetrisNet | None = None,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)

        # Validate paths are set (should be done by setup_run_directory)
        if config.run.checkpoint_dir is None or config.run.data_dir is None:
            raise ValueError(
                "checkpoint_dir and data_dir must be set. "
                "Call setup_run_directory() before creating Trainer."
            )

        # Create model
        if model is None:
            model = TetrisNet(
                trunk_channels=config.network.trunk_channels,
                num_conv_residual_blocks=config.network.num_conv_residual_blocks,
                reduction_channels=config.network.reduction_channels,
                fc_hidden=config.network.fc_hidden,
                conv_kernel_size=config.network.conv_kernel_size,
                conv_padding=config.network.conv_padding,
            )
        self.model = model.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create weight manager
        self.weight_manager = WeightManager(config.run.checkpoint_dir)

        # Training state
        self.step = 0
        self.loss_balancer = RunningLossBalancer(
            config.optimizer.value_loss_weight_window
        )
        self._cached_value_loss_weight: float = 1.0
        self._pending_eval_gif_paths: list[Path] = []
        self.initial_incumbent_model_path: Path | None = None
        self.initial_incumbent_eval_avg_attack: float = 0.0

        # Create directories
        config.run.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.run.data_dir.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self):
        if self.config.optimizer.lr_schedule == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.optimizer.lr_min_factor,
                total_iters=self.config.optimizer.lr_decay_steps,
            )
        elif self.config.optimizer.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.optimizer.lr_decay_steps,
                eta_min=self.config.optimizer.learning_rate
                * self.config.optimizer.lr_min_factor,
            )
        elif self.config.optimizer.lr_schedule == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.optimizer.lr_decay_steps
                // self.config.optimizer.lr_step_divisor,
                gamma=self.config.optimizer.lr_step_gamma,
            )
        else:
            return None

    def align_scheduler_to_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0 (got {step})")
        if self.scheduler is None:
            return

        # Rebuild scheduler state from current config so resumed runs use the new
        # LR settings while keeping global step alignment.
        self.scheduler.last_epoch = step

        if self.config.optimizer.lr_schedule == "linear":
            assert isinstance(self.scheduler, torch.optim.lr_scheduler.LinearLR)
            total_iters = self.config.optimizer.lr_decay_steps
            progress = min(step, total_iters) / total_iters
            factor = 1.0 + (self.config.optimizer.lr_min_factor - 1.0) * progress
            lrs = [base_lr * factor for base_lr in self.scheduler.base_lrs]
        elif self.config.optimizer.lr_schedule == "cosine":
            assert isinstance(
                self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            )
            t_max = self.config.optimizer.lr_decay_steps
            eta_min = (
                self.config.optimizer.learning_rate
                * self.config.optimizer.lr_min_factor
            )
            cosine_factor = (
                1 + torch.cos(torch.tensor(torch.pi * step / t_max))
            ).item() / 2
            lrs = [
                eta_min + (base_lr - eta_min) * cosine_factor
                for base_lr in self.scheduler.base_lrs
            ]
        elif self.config.optimizer.lr_schedule == "step":
            assert isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR)
            step_size = (
                self.config.optimizer.lr_decay_steps
                // self.config.optimizer.lr_step_divisor
            )
            decay = self.config.optimizer.lr_step_gamma ** (step // step_size)
            lrs = [base_lr * decay for base_lr in self.scheduler.base_lrs]
        else:
            raise ValueError(
                f"Unsupported lr_schedule: {self.config.optimizer.lr_schedule}"
            )

        if len(self.optimizer.param_groups) != len(lrs):
            raise ValueError(
                "Optimizer param group count does not match computed LR count"
            )
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self.scheduler._last_lr = lrs

    @staticmethod
    def _compute_candidate_nn_value_weight(
        current_weight: float,
        config: TrainingConfig,
    ) -> float:
        if current_weight < 0.0:
            raise ValueError(f"current_weight must be >= 0 (got {current_weight})")
        promotion_delta = current_weight * (
            config.self_play.nn_value_weight_promotion_multiplier - 1.0
        )
        delta = min(
            promotion_delta,
            config.self_play.nn_value_weight_promotion_max_delta,
        )
        return min(config.self_play.nn_value_weight_cap, current_weight + delta)

    def _build_training_batch(
        self,
        sample: tuple,
    ) -> TrainingBatch:
        (
            boards,
            aux,
            policy_targets,
            value_targets,
            overhang_fields,
            masks,
        ) = sample
        return TrainingBatch(
            boards=torch.from_numpy(boards).reshape(-1, 1, BOARD_HEIGHT, BOARD_WIDTH),
            aux=torch.from_numpy(aux),
            policy_targets=torch.from_numpy(policy_targets),
            value_targets=torch.from_numpy(value_targets),
            overhang_fields=torch.from_numpy(overhang_fields),
            masks=torch.from_numpy(masks),
        )

    def _pin_batch_if_needed(self, batch: TrainingBatch) -> TrainingBatch:
        should_pin = (
            self.device.type == "cuda"
            and self.config.replay.pin_memory_batches
            and batch.device.type == "cpu"
        )
        if not should_pin:
            return batch
        return TrainingBatch(
            boards=batch.boards.pin_memory(),
            aux=batch.aux.pin_memory(),
            policy_targets=batch.policy_targets.pin_memory(),
            value_targets=batch.value_targets.pin_memory(),
            overhang_fields=batch.overhang_fields.pin_memory(),
            masks=batch.masks.pin_memory(),
        )

    def _is_batch_on_training_device(self, batch: TrainingBatch) -> bool:
        batch_device = batch.device
        if batch_device.type != self.device.type:
            return False
        if self.device.type != "cuda":
            return True
        if self.device.index is None:
            return True
        return batch_device.index == self.device.index

    def _to_training_device(self, batch: TrainingBatch) -> TrainingBatch:
        if self._is_batch_on_training_device(batch):
            return batch
        batch = self._pin_batch_if_needed(batch)
        non_blocking = self.device.type == "cuda"
        return TrainingBatch(
            boards=batch.boards.to(self.device, non_blocking=non_blocking),
            aux=batch.aux.to(self.device, non_blocking=non_blocking),
            policy_targets=batch.policy_targets.to(
                self.device, non_blocking=non_blocking
            ),
            value_targets=batch.value_targets.to(
                self.device, non_blocking=non_blocking
            ),
            overhang_fields=batch.overhang_fields.to(
                self.device, non_blocking=non_blocking
            ),
            masks=batch.masks.to(self.device, non_blocking=non_blocking),
        )

    def _training_batch_bytes(self, batch: TrainingBatch) -> int:
        tensors = (
            batch.boards,
            batch.aux,
            batch.policy_targets,
            batch.value_targets,
            batch.overhang_fields,
            batch.masks,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @staticmethod
    def _tensor_field_pairs(
        mirror: CircularReplayMirror,
        batch: TrainingBatch,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (mirror.boards, batch.boards),
            (mirror.aux, batch.aux),
            (mirror.policy_targets, batch.policy_targets),
            (mirror.value_targets, batch.value_targets),
            (mirror.overhang_fields, batch.overhang_fields),
            (mirror.masks, batch.masks),
        ]

    def _write_to_mirror(
        self,
        mirror: CircularReplayMirror,
        batch: TrainingBatch,
    ) -> None:
        n = batch.size
        if n == 0:
            return
        if n > mirror.capacity:
            offset = n - mirror.capacity
            batch = TrainingBatch(
                boards=batch.boards[offset:],
                aux=batch.aux[offset:],
                policy_targets=batch.policy_targets[offset:],
                value_targets=batch.value_targets[offset:],
                overhang_fields=batch.overhang_fields[offset:],
                masks=batch.masks[offset:],
            )
            n = mirror.capacity

        end_pos = mirror.write_pos + n
        pairs = self._tensor_field_pairs(mirror, batch)
        if end_pos <= mirror.capacity:
            s = slice(mirror.write_pos, end_pos)
            for dst, src in pairs:
                dst[s].copy_(src)
        else:
            tail = mirror.capacity - mirror.write_pos
            head = n - tail
            for dst, src in pairs:
                dst[mirror.write_pos :].copy_(src[:tail])
                dst[:head].copy_(src[tail:])

        mirror.write_pos = (mirror.write_pos + n) % mirror.capacity
        mirror.count = min(mirror.count + n, mirror.capacity)

    def _sample_prefetched_batches(
        self,
        generator: GameGenerator,
        staged_batch_size: int,
    ) -> list[TrainingBatch] | None:
        result = generator.sample_batch(
            staged_batch_size, self.config.self_play.max_placements
        )
        if result is None:
            return None
        staged_batch = self._to_training_device(self._build_training_batch(result))
        return staged_batch.split(self.config.optimizer.batch_size)

    def _use_device_replay_mirror(self) -> bool:
        return (
            self.config.replay.mirror_replay_on_accelerator
            and self.device.type != "cpu"
        )

    def _load_replay_mirror(
        self,
        generator: GameGenerator,
        mirror: CircularReplayMirror | None = None,
    ) -> CircularReplayMirror | None:
        result = generator.replay_buffer_snapshot(self.config.self_play.max_placements)
        if result is None:
            return None
        (
            start_index,
            boards,
            aux,
            policy_targets,
            value_targets,
            overhang_fields,
            masks,
        ) = result
        device_batch = self._to_training_device(
            self._build_training_batch(
                (boards, aux, policy_targets, value_targets, overhang_fields, masks)
            )
        )
        if mirror is None:
            mirror = CircularReplayMirror(self.config.replay.buffer_size, self.device)
        mirror.count = 0
        mirror.write_pos = 0
        self._write_to_mirror(mirror, device_batch)
        mirror.logical_end = int(start_index) + device_batch.size
        logger.info(
            "Loaded replay mirror snapshot",
            start_index=int(start_index),
            examples=mirror.count,
            device=str(self.device),
        )
        return mirror

    def _refresh_replay_mirror(
        self,
        generator: GameGenerator,
        mirror: CircularReplayMirror | None,
    ) -> CircularReplayMirror | None:
        if mirror is None:
            return self._load_replay_mirror(generator)

        delta_examples_total = 0
        delta_bytes_total = 0
        while True:
            result = generator.replay_buffer_delta(
                mirror.logical_end,
                self.config.replay.replay_mirror_delta_chunk_examples,
                self.config.self_play.max_placements,
            )
            if result is None:
                return None
            (
                window_start,
                window_end,
                slice_start,
                boards,
                aux,
                policy_targets,
                value_targets,
                overhang_fields,
                masks,
            ) = result
            window_start = int(window_start)
            window_end = int(window_end)
            slice_start = int(slice_start)
            if slice_start < mirror.logical_end:
                logger.info(
                    "Replay mirror stale; loading full snapshot",
                    mirror_logical_end=mirror.logical_end,
                    window_start_index=window_start,
                    window_end_index=window_end,
                    slice_start_index=slice_start,
                )
                return self._load_replay_mirror(generator, mirror)
            if window_start > mirror.logical_end:
                logger.info(
                    "Replay mirror fully evicted; rebasing",
                    mirror_logical_end=mirror.logical_end,
                    window_start_index=window_start,
                    window_end_index=window_end,
                )
                mirror.count = 0
                mirror.write_pos = 0
                mirror.logical_end = window_start

            delta_batch = self._to_training_device(
                self._build_training_batch(
                    (boards, aux, policy_targets, value_targets, overhang_fields, masks)
                )
            )
            if delta_batch.size > 0:
                delta_examples_total += delta_batch.size
                delta_bytes_total += self._training_batch_bytes(delta_batch)
                self._write_to_mirror(mirror, delta_batch)
                mirror.logical_end += delta_batch.size

            if mirror.logical_end >= window_end:
                if delta_examples_total > 0:
                    logger.info(
                        "Updated replay mirror incrementally",
                        added_examples=delta_examples_total,
                        delta_gb=(delta_bytes_total / (1024.0 * 1024.0 * 1024.0)),
                        mirror_logical_end=mirror.logical_end,
                        mirror_examples=mirror.count,
                        window_start_index=window_start,
                        window_end_index=window_end,
                    )
                return mirror

    def _sample_from_replay_mirror(self, mirror: CircularReplayMirror) -> TrainingBatch:
        if mirror.count <= 0:
            raise ValueError("Replay mirror is empty")
        sample_indices = torch.randint(
            low=0,
            high=mirror.count,
            size=(self.config.optimizer.batch_size,),
            device=mirror.boards.device,
        )
        return TrainingBatch(
            boards=mirror.boards.index_select(0, sample_indices),
            aux=mirror.aux.index_select(0, sample_indices),
            policy_targets=mirror.policy_targets.index_select(0, sample_indices),
            value_targets=mirror.value_targets.index_select(0, sample_indices),
            overhang_fields=mirror.overhang_fields.index_select(0, sample_indices),
            masks=mirror.masks.index_select(0, sample_indices),
        )

    def train_step(
        self, batch: TrainingBatch, collect_metrics: bool
    ) -> dict[str, float]:
        """Execute one training step."""
        self.model.train()

        batch = self._to_training_device(batch)
        boards = batch.boards
        aux = batch.aux
        policy_targets = batch.policy_targets
        value_targets = batch.value_targets
        overhang_fields = batch.overhang_fields
        masks = batch.masks

        # Forward + backward
        self.optimizer.zero_grad(set_to_none=True)
        value_loss_weight = self._cached_value_loss_weight
        total_loss, policy_loss, value_loss = compute_loss(
            self.model,
            boards,
            aux,
            policy_targets,
            value_targets,
            masks,
            value_loss_weight,
            self.config.optimizer.use_huber_value_loss,
        )
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.optimizer.grad_clip_norm
        )

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        if not collect_metrics:
            return {}

        # Only sync GPU and update loss balancer when collecting metrics
        policy_loss_scalar = policy_loss.item()
        value_loss_scalar = value_loss.item()
        self.loss_balancer.append(policy_loss_scalar, value_loss_scalar)
        if self.loss_balancer.has_history():
            self._cached_value_loss_weight = self.loss_balancer.value_loss_weight()

        policy_loss_avg, value_loss_avg = self.loss_balancer.averages()
        metrics = {
            "train/loss": total_loss.item(),
            "train/policy_loss": policy_loss_scalar,
            "train/value_loss": value_loss_scalar,
            "train/policy_loss_avg": policy_loss_avg,
            "train/value_loss_avg": value_loss_avg,
            "train/value_loss_weight": value_loss_weight,
            "train/grad_norm": grad_norm.item(),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        metrics.update(
            compute_batch_feature_metrics(
                boards=boards,
                aux=aux,
                value_targets=value_targets,
                overhang_fields=overhang_fields,
                masks=masks,
            )
        )
        return metrics

    def _compute_extra_train_metrics(self, batch: TrainingBatch) -> dict:
        batch = self._to_training_device(batch)
        return compute_metrics(
            self.model,
            batch.boards,
            batch.aux,
            batch.policy_targets,
            batch.value_targets,
            batch.masks,
        )

    def _persist_incumbent_model_artifacts(
        self, generator: GameGenerator
    ) -> tuple[Path | None, str]:
        if not generator.incumbent_uses_network():
            source_path_string = generator.incumbent_model_path()
            return None, source_path_string
        if self.config.run.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is not set on training config")
        destination_path = self.config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
        source_path_string = ""
        for attempt in range(2):
            source_path = Path(generator.incumbent_model_path())
            source_path_string = str(source_path)
            try:
                copy_model_artifact_bundle(source_path, destination_path)
                return destination_path, source_path_string
            except (FileNotFoundError, RuntimeError) as error:
                latest_source_path = Path(generator.incumbent_model_path())
                if attempt == 0 and latest_source_path != source_path:
                    logger.warning(
                        "Incumbent artifact changed while checkpointing; retrying copy",
                        source_path=source_path_string,
                        latest_source_path=str(latest_source_path),
                    )
                    continue
                raise RuntimeError(
                    "Failed to persist incumbent model artifacts for checkpoint "
                    f"(source={source_path_string})"
                ) from error
        raise RuntimeError(
            "Failed to persist incumbent model artifacts after retry "
            f"(source={source_path_string})"
        )

    def _create_wandb_gif_video(
        self,
        frames: list,
        attack: int,
    ) -> tuple[wandb.Video | None, Path | None]:
        if not frames:
            return None, None

        gif_path = (
            Path(tempfile.gettempdir()) / f"eval_step{self.step}_attack{attack}.gif"
        )

        try:
            create_trajectory_gif(
                frames=frames,
                output_path=str(gif_path),
                duration=DEFAULT_GIF_FRAME_DURATION_MS,
            )

            video = wandb.Video(
                str(gif_path),
                format="gif",
            )
            self._pending_eval_gif_paths.append(gif_path)
            return video, gif_path
        except BaseException:
            gif_path.unlink(missing_ok=True)
            raise

    def _cleanup_wandb_gif_files(self) -> None:
        for gif_path in self._pending_eval_gif_paths:
            try:
                gif_path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning(
                    "Failed to delete temporary eval GIF",
                    path=str(gif_path),
                    error=str(error),
                )
        self._pending_eval_gif_paths.clear()

    def save(self, extra_checkpoint_state: dict[str, object] | None = None):
        """Save model checkpoint."""
        paths = self.weight_manager.save(
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            export_for_rust=True,
            extra_checkpoint_state=extra_checkpoint_state,
        )
        logger.info(
            "Saved checkpoint",
            step=self.step,
            checkpoint=str(paths["checkpoint"]),
            onnx=str(paths.get("onnx")) if "onnx" in paths else None,
        )
        return paths

    def _log_final_wandb_model_artifact(self, saved_paths: dict[str, Path]) -> None:
        if wandb.run is None:
            logger.warning(
                "Skipping final WandB model artifact upload; no active WandB run",
                step=self.step,
            )
            return

        artifact_name = f"tetris-model-{wandb.run.id}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description="Final model snapshot saved when training loop stops",
            metadata={
                "trainer_step": self.step,
                "run_name": self.config.run.run_name,
            },
        )

        files_to_upload = [saved_paths["checkpoint"], saved_paths["metadata"]]
        onnx_path = saved_paths.get("onnx")
        if onnx_path is not None:
            files_to_upload.append(onnx_path)
            conv_path, heads_path, fc_path = split_model_paths(onnx_path)
            files_to_upload.extend([conv_path, heads_path, fc_path])
        else:
            # WeightManager.save currently always exports ONNX for Rust; fail fast if this changes.
            checkpoint_dir = self.config.run.checkpoint_dir
            if checkpoint_dir is None:
                raise RuntimeError("checkpoint_dir is not set on training config")
            expected_onnx_path = checkpoint_dir / LATEST_ONNX_FILENAME
            raise RuntimeError(
                "Saved paths missing ONNX artifact during final WandB upload "
                f"(expected {expected_onnx_path})"
            )

        checkpoint_dir = self.config.run.checkpoint_dir
        if checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is not set on training config")
        incumbent_onnx_path = checkpoint_dir / INCUMBENT_ONNX_FILENAME
        if incumbent_onnx_path.exists():
            assert_rust_inference_artifacts(incumbent_onnx_path)
            files_to_upload.append(incumbent_onnx_path)
            incumbent_conv_path, incumbent_heads_path, incumbent_fc_path = (
                split_model_paths(incumbent_onnx_path)
            )
            files_to_upload.extend(
                [incumbent_conv_path, incumbent_heads_path, incumbent_fc_path]
            )

        data_dir = self.config.run.data_dir
        if data_dir is None:
            raise RuntimeError("data_dir is not set on training config")
        training_data_path = data_dir / TRAINING_DATA_FILENAME
        if training_data_path.exists():
            files_to_upload.append(training_data_path)

        for file_path in files_to_upload:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Expected model artifact file is missing: {file_path}"
                )
            artifact.add_file(str(file_path), name=file_path.name)

        wandb.log_artifact(artifact, aliases=["latest", "final", f"step-{self.step}"])
        logger.info(
            "Uploaded final model artifact to WandB",
            artifact_name=artifact_name,
            step=self.step,
            files=[path.name for path in files_to_upload],
        )

    def train(self, log_to_wandb: bool = True):
        """
        Run parallel training with Rust game generation in background.

        The Rust GameGenerator runs in a background thread, continuously
        generating games into a shared in-memory buffer. Python samples
        directly from the buffer via generator.sample_batch(). At regular
        wall-clock intervals, a new ONNX model is exported for the
        generator to pick up.

        Args:
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        num_steps = self.config.optimizer.total_steps
        if self.step >= num_steps:
            logger.info(
                "Target step already reached; skipping training loop",
                target_step=num_steps,
                current_step=self.step,
            )
            return
        # Paths for parallel training (validated in __init__)
        assert self.config.run.checkpoint_dir is not None
        assert self.config.run.data_dir is not None
        onnx_path = self.config.run.checkpoint_dir / PARALLEL_ONNX_FILENAME
        candidate_model_dir = self.config.run.checkpoint_dir / MODEL_CANDIDATES_DIRNAME
        candidate_model_dir.mkdir(parents=True, exist_ok=True)

        # Export initial model (full ONNX + split models for cached Rust inference)
        full_export_ok = export_onnx(self.model, onnx_path)
        split_export_ok = export_split_models(self.model, onnx_path)
        if not full_export_ok:
            raise RuntimeError("ONNX export failed due to missing dependencies")
        if not split_export_ok:
            raise RuntimeError("Split-model export failed due to missing dependencies")
        assert_rust_inference_artifacts(onnx_path)

        # Optionally compile model for faster training forward/backward
        export_model = self.model
        if self.config.optimizer.use_torch_compile:
            logger.info("Compiling model with torch.compile")
            self.model = cast(TetrisNet, torch.compile(self.model))

        generator_model_path = onnx_path
        if self.initial_incumbent_model_path is not None:
            assert_rust_inference_artifacts(self.initial_incumbent_model_path)
            generator_model_path = self.initial_incumbent_model_path

        # Create MCTS config for generator
        mcts_config = MCTSConfig()
        mcts_config.self_play.num_simulations = self.config.self_play.num_simulations
        mcts_config.self_play.c_puct = self.config.self_play.c_puct
        mcts_config.self_play.temperature = self.config.self_play.temperature
        mcts_config.self_play.dirichlet_alpha = self.config.self_play.dirichlet_alpha
        mcts_config.self_play.dirichlet_epsilon = (
            self.config.self_play.dirichlet_epsilon
        )
        mcts_config.self_play.visit_sampling_epsilon = (
            self.config.self_play.visit_sampling_epsilon
        )
        mcts_config.self_play.max_placements = self.config.self_play.max_placements
        mcts_config.self_play.death_penalty = self.config.self_play.death_penalty
        mcts_config.self_play.overhang_penalty_weight = (
            self.config.self_play.overhang_penalty_weight
        )
        mcts_config.self_play.nn_value_weight = self.config.self_play.nn_value_weight
        mcts_config.self_play.q_scale = (
            self.config.self_play.q_scale
            if self.config.self_play.use_tanh_q_normalization
            else None
        )
        mcts_config.self_play.reuse_tree = self.config.self_play.reuse_tree

        # Start background game generator
        training_data_path = self.config.run.data_dir / TRAINING_DATA_FILENAME
        generator = GameGenerator(
            model_path=str(generator_model_path),
            training_data_path=str(training_data_path),
            config=mcts_config,
            max_placements=self.config.self_play.max_placements,
            add_noise=self.config.self_play.add_noise,
            max_examples=self.config.replay.buffer_size,
            save_interval_seconds=self.config.run.save_interval_seconds,
            num_workers=self.config.self_play.num_workers,
            initial_model_step=self.step,
            candidate_eval_seeds=list(
                range(self.config.self_play.model_promotion_eval_games)
            ),
            start_with_network=not self.config.self_play.bootstrap_without_network,
            non_network_num_simulations=self.config.self_play.bootstrap_num_simulations,
            initial_incumbent_eval_avg_attack=self.initial_incumbent_eval_avg_attack,
            nn_value_weight_cap=self.config.self_play.nn_value_weight_cap,
        )
        generator.start()
        logger.info(
            "Started background game generator",
            model_path=str(generator_model_path),
            trainer_parallel_model_path=str(onnx_path),
            training_data_path=str(training_data_path),
            num_workers=self.config.self_play.num_workers,
            add_noise=self.config.self_play.add_noise,
            candidate_eval_seeds=self.config.self_play.model_promotion_eval_games,
            bootstrap_without_network=self.config.self_play.bootstrap_without_network,
            bootstrap_num_simulations=self.config.self_play.bootstrap_num_simulations,
            incumbent_nn_value_weight=self.config.self_play.nn_value_weight,
            initial_incumbent_eval_avg_attack=self.initial_incumbent_eval_avg_attack,
            nn_value_weight_promotion_multiplier=self.config.self_play.nn_value_weight_promotion_multiplier,
            nn_value_weight_promotion_max_delta=self.config.self_play.nn_value_weight_promotion_max_delta,
            nn_value_weight_cap=self.config.self_play.nn_value_weight_cap,
        )

        # Wait for minimum buffer size
        logger.info(
            "Waiting for minimum replay buffer size",
            min_examples=self.config.replay.min_buffer_size,
        )
        while generator.buffer_size() < self.config.replay.min_buffer_size:
            time.sleep(1.0)
            logger.info(
                "Buffer fill progress",
                buffer_size=generator.buffer_size(),
                games_generated=generator.games_generated(),
            )

        start_step = self.step
        logger.info(
            "Starting training loop",
            target_step=num_steps,
            start_step=start_step,
            config=str(self.config),
        )
        use_device_replay_mirror = self._use_device_replay_mirror()
        staged_batch_size = (
            self.config.optimizer.batch_size * self.config.replay.prefetch_batches
        )
        staged_queue_target_batches = self.config.replay.staged_batch_cache_batches
        if use_device_replay_mirror:
            logger.info(
                "Configured full replay device mirroring",
                device=str(self.device),
                train_batch_size=self.config.optimizer.batch_size,
                refresh_seconds=self.config.replay.replay_mirror_refresh_seconds,
                delta_chunk_examples=self.config.replay.replay_mirror_delta_chunk_examples,
                pin_memory_batches=(
                    self.config.replay.pin_memory_batches and self.device.type == "cuda"
                ),
            )
        else:
            logger.info(
                "Configured staged replay sampling",
                train_batch_size=self.config.optimizer.batch_size,
                prefetch_batches=self.config.replay.prefetch_batches,
                staged_batch_size=staged_batch_size,
                staged_queue_target_batches=staged_queue_target_batches,
                device=str(self.device),
                pin_memory_batches=(
                    self.config.replay.pin_memory_batches and self.device.type == "cuda"
                ),
            )

        sample_batch_time_s = 0.0
        sample_batch_count = 0
        replay_sync_time_s = 0.0
        replay_sync_count = 0
        train_step_time_s = 0.0
        train_step_count = 0
        latest_train_metrics: dict[str, float] | None = None
        pending_batches: deque[TrainingBatch] = deque()
        replay_mirror: CircularReplayMirror | None = None
        interval_anchor_s = time.perf_counter()
        throughput_window_start_s = interval_anchor_s
        throughput_window_start_games = generator.games_generated()
        throughput_window_start_steps = 0
        next_log_time_s = interval_anchor_s + self.config.run.log_interval_seconds
        next_replay_sync_time_s = interval_anchor_s
        next_model_sync_time_s = (
            interval_anchor_s + self.config.run.model_sync_interval_seconds
        )
        next_checkpoint_time_s = (
            interval_anchor_s + self.config.run.checkpoint_interval_seconds
        )

        interrupted = False
        pending_error: BaseException | None = None
        stop_error: BaseException | None = None

        try:
            while self.step < num_steps:
                pre_step_time = time.perf_counter()

                if use_device_replay_mirror:
                    should_refresh_mirror = (
                        replay_mirror is None
                        or pre_step_time >= next_replay_sync_time_s
                    )
                    if should_refresh_mirror:
                        replay_sync_start = time.perf_counter()
                        replay_mirror = self._refresh_replay_mirror(
                            generator, replay_mirror
                        )
                        replay_sync_elapsed_s = time.perf_counter() - replay_sync_start
                        if replay_mirror is None:
                            time.sleep(0.1)
                            continue
                        replay_sync_time_s += replay_sync_elapsed_s
                        replay_sync_count += 1
                        next_replay_sync_time_s = (
                            pre_step_time
                            + self.config.replay.replay_mirror_refresh_seconds
                        )
                    if replay_mirror is None:
                        raise RuntimeError(
                            "Replay mirror mode active but replay mirror is unavailable"
                        )
                    batch = self._sample_from_replay_mirror(replay_mirror)
                else:
                    should_refill = len(pending_batches) < staged_queue_target_batches
                    if should_refill:
                        sample_batch_start = time.perf_counter()
                        prefetched_batches = self._sample_prefetched_batches(
                            generator=generator,
                            staged_batch_size=staged_batch_size,
                        )
                        sample_batch_elapsed_s = (
                            time.perf_counter() - sample_batch_start
                        )
                        if prefetched_batches is None:
                            if not pending_batches:
                                time.sleep(0.1)
                                continue
                        else:
                            pending_batches.extend(prefetched_batches)
                            sample_batch_time_s += sample_batch_elapsed_s
                            sample_batch_count += len(prefetched_batches)

                    batch = pending_batches.popleft()
                self.step += 1
                session_step = self.step - start_step

                # Train step
                is_log_step = pre_step_time >= next_log_time_s
                collect_train_metrics = (
                    is_log_step
                    or latest_train_metrics is None
                    or session_step % self.config.optimizer.train_step_metrics_interval
                    == 0
                )
                step_metrics = self.train_step(
                    batch,
                    collect_metrics=collect_train_metrics,
                )
                post_step_time = time.perf_counter()
                train_step_time_s += post_step_time - pre_step_time
                train_step_count += 1
                if step_metrics:
                    latest_train_metrics = step_metrics

                # Log metrics + drain eval events (at log-tick frequency)
                if post_step_time >= next_log_time_s:
                    for event in generator.drain_model_eval_events():
                        promoted = bool(event["promoted"])
                        candidate_nn_value_weight = float(
                            event["candidate_nn_value_weight"]
                        )
                        incumbent_nn_value_weight = float(
                            event["incumbent_nn_value_weight"]
                        )
                        promoted_nn_value_weight = float(
                            event["promoted_nn_value_weight"]
                        )
                        promoted_death_penalty = float(event["promoted_death_penalty"])
                        promoted_overhang_penalty_weight = float(
                            event["promoted_overhang_penalty_weight"]
                        )
                        logger.info(
                            "Model evaluation decision",
                            trainer_step=self.step,
                            candidate_step=int(event["candidate_step"]),
                            candidate_games=int(event["candidate_games"]),
                            candidate_avg_attack=event["candidate_avg_attack"],
                            candidate_attack_variance=event[
                                "candidate_attack_variance"
                            ],
                            candidate_nn_value_weight=candidate_nn_value_weight,
                            incumbent_step=int(event["incumbent_step"]),
                            incumbent_uses_network=bool(
                                event["incumbent_uses_network"]
                            ),
                            incumbent_avg_attack=event["incumbent_avg_attack"],
                            incumbent_nn_value_weight=incumbent_nn_value_weight,
                            promoted_nn_value_weight=promoted_nn_value_weight,
                            promoted_death_penalty=promoted_death_penalty,
                            promoted_overhang_penalty_weight=promoted_overhang_penalty_weight,
                            promoted=promoted,
                            auto_promoted=bool(event["auto_promoted"]),
                            evaluation_seconds=event["evaluation_seconds"],
                        )
                        if log_to_wandb:
                            wandb_data: dict[str, object] = {
                                "trainer_step": self.step,
                                "model_gate/candidate_step": event["candidate_step"],
                                "model_gate/candidate_games": event["candidate_games"],
                                "model_gate/candidate_avg_attack": event[
                                    "candidate_avg_attack"
                                ],
                                "model_gate/candidate_attack_variance": event[
                                    "candidate_attack_variance"
                                ],
                                "model_gate/candidate_nn_value_weight": candidate_nn_value_weight,
                                "model_gate/incumbent_step": event["incumbent_step"],
                                "model_gate/incumbent_uses_network": event[
                                    "incumbent_uses_network"
                                ],
                                "model_gate/incumbent_avg_attack": event[
                                    "incumbent_avg_attack"
                                ],
                                "model_gate/incumbent_nn_value_weight": incumbent_nn_value_weight,
                                "model_gate/promoted_nn_value_weight": promoted_nn_value_weight,
                                "model_gate/promoted_death_penalty": promoted_death_penalty,
                                "model_gate/promoted_overhang_penalty_weight": promoted_overhang_penalty_weight,
                                "model_gate/promoted": event["promoted"],
                                "model_gate/auto_promoted": event["auto_promoted"],
                                "model_gate/evaluation_seconds": event[
                                    "evaluation_seconds"
                                ],
                            }

                            # Compute eval/* metrics from per-game results
                            per_game_results = event["per_game_results"]
                            if per_game_results:
                                num_games = len(per_game_results)
                                total_attack = sum(r[1] for r in per_game_results)
                                total_lines = sum(r[2] for r in per_game_results)
                                total_moves = sum(r[3] for r in per_game_results)
                                wandb_data["eval/num_games"] = num_games
                                wandb_data["eval/avg_attack"] = total_attack / num_games
                                wandb_data["eval/max_attack"] = max(
                                    r[1] for r in per_game_results
                                )
                                wandb_data["eval/avg_lines"] = total_lines / num_games
                                wandb_data["eval/max_lines"] = max(
                                    r[2] for r in per_game_results
                                )
                                wandb_data["eval/avg_moves"] = total_moves / num_games
                                wandb_data["eval/attack_per_piece"] = (
                                    total_attack / total_moves
                                    if total_moves > 0
                                    else 0.0
                                )
                                wandb_data["eval/lines_per_piece"] = (
                                    total_lines / total_moves
                                    if total_moves > 0
                                    else 0.0
                                )
                                wandb_data["eval/nn_value_weight"] = (
                                    promoted_nn_value_weight
                                )

                            # Render best/worst trajectory GIFs
                            best_replay = event.get("best_game_replay")
                            worst_replay = event.get("worst_game_replay")
                            if best_replay is not None:
                                frames = render_replay(best_replay)
                                video, _ = self._create_wandb_gif_video(
                                    frames, attack=best_replay.total_attack
                                )
                                if video is not None:
                                    wandb_data["eval/best_trajectory"] = video
                            if worst_replay is not None:
                                frames = render_replay(worst_replay)
                                video, _ = self._create_wandb_gif_video(
                                    frames, attack=worst_replay.total_attack
                                )
                                if video is not None:
                                    wandb_data["eval/worst_trajectory"] = video

                            wandb.log(wandb_data)

                    if latest_train_metrics is None:
                        raise RuntimeError(
                            "No collected train metrics are available for logging"
                        )
                    metrics = dict(latest_train_metrics)
                    if self.config.optimizer.compute_extra_train_metrics_on_log:
                        extra_metrics_start = time.perf_counter()
                        metrics.update(self._compute_extra_train_metrics(batch))
                        metrics["timing/extra_metrics_ms"] = 1000.0 * (
                            time.perf_counter() - extra_metrics_start
                        )
                    games = generator.games_generated()
                    window_elapsed_s = post_step_time - throughput_window_start_s
                    games_delta = games - throughput_window_start_games
                    steps_delta = session_step - throughput_window_start_steps
                    metrics["replay/buffer_size"] = generator.buffer_size()
                    metrics["replay/games_generated"] = games
                    metrics["replay/examples_generated"] = (
                        generator.examples_generated()
                    )
                    metrics["replay/source"] = 1.0 if use_device_replay_mirror else 0.0
                    if use_device_replay_mirror:
                        if replay_mirror is None:
                            raise RuntimeError(
                                "Replay mirror mode active but replay_mirror is missing"
                            )
                        metrics["replay/mirror_size"] = replay_mirror.size
                        metrics["replay/mirror_logical_end"] = replay_mirror.logical_end
                    metrics["incumbent/model_step"] = generator.incumbent_model_step()
                    metrics["incumbent/uses_network"] = (
                        generator.incumbent_uses_network()
                    )
                    metrics["incumbent/nn_value_weight"] = (
                        generator.incumbent_nn_value_weight()
                    )
                    metrics["throughput/games_per_second"] = (
                        games_delta / window_elapsed_s if window_elapsed_s > 0 else 0.0
                    )
                    metrics["throughput/steps_per_second"] = (
                        steps_delta / window_elapsed_s if window_elapsed_s > 0 else 0.0
                    )
                    throughput_window_start_s = post_step_time
                    throughput_window_start_games = games
                    throughput_window_start_steps = session_step
                    metrics["timing/sample_batch_ms"] = (
                        1000.0 * sample_batch_time_s / sample_batch_count
                        if sample_batch_count > 0 and not use_device_replay_mirror
                        else 0.0
                    )
                    metrics["timing/replay_sync_ms"] = (
                        1000.0 * replay_sync_time_s / replay_sync_count
                        if replay_sync_count > 0 and use_device_replay_mirror
                        else 0.0
                    )
                    metrics["timing/train_step_ms"] = (
                        1000.0 * train_step_time_s / train_step_count
                        if train_step_count > 0
                        else 0.0
                    )
                    if log_to_wandb:
                        metrics["trainer_step"] = self.step
                        wandb.log(metrics)
                        completed_games = generator.drain_completed_game_stats()
                        if self.config.optimizer.log_individual_games_to_wandb:
                            for (
                                game_number,
                                game_stats,
                            ) in completed_games:
                                game_metrics = {
                                    "game_number": game_number,
                                    "game/number": game_number,
                                    "trainer_step": self.step,
                                }
                                for key, value in game_stats.items():
                                    game_metrics[f"game/{key}"] = value
                                episode_length = game_stats["episode_length"]
                                if episode_length <= 0:
                                    raise ValueError(
                                        "Invalid episode_length for game "
                                        f"{game_number}: {episode_length}"
                                    )
                                game_metrics["game/attack_per_move"] = (
                                    game_stats["total_attack"] / episode_length
                                )
                                game_metrics["game/lines_per_move"] = (
                                    game_stats["total_lines"] / episode_length
                                )
                                game_metrics["game/hold_rate"] = (
                                    game_stats["holds"] / episode_length
                                )
                                # Don't pin per-game logs to the training step:
                                # multiple games can complete between train
                                # ticks, and reusing the same step causes only
                                # a subset to appear in history.
                                wandb.log(game_metrics)
                        else:
                            game_summary_metrics = summarize_completed_games(
                                completed_games
                            )
                            if game_summary_metrics:
                                game_summary_metrics["trainer_step"] = self.step
                                wandb.log(game_summary_metrics)
                    sample_batch_time_s = 0.0
                    sample_batch_count = 0
                    replay_sync_time_s = 0.0
                    replay_sync_count = 0
                    train_step_time_s = 0.0
                    train_step_count = 0
                    logger.info(
                        "Training progress",
                        step=self.step,
                        loss=metrics["train/loss"],
                        learning_rate=metrics["train/learning_rate"],
                        buffer_size=generator.buffer_size(),
                        games_generated=games,
                        games_per_second=metrics["throughput/games_per_second"],
                        steps_per_second=metrics["throughput/steps_per_second"],
                        sample_batch_ms=metrics["timing/sample_batch_ms"],
                        replay_sync_ms=metrics["timing/replay_sync_ms"],
                        train_step_ms=metrics["timing/train_step_ms"],
                    )
                    next_log_time_s = roll_interval_deadline(
                        next_log_time_s,
                        self.config.run.log_interval_seconds,
                        post_step_time,
                    )

                # Export updated model for generator
                if post_step_time >= next_model_sync_time_s:
                    candidate_onnx_path = (
                        candidate_model_dir / f"candidate_step_{self.step}.onnx"
                    )
                    onnx_export_start = time.perf_counter()
                    full_export_ok = export_onnx(export_model, candidate_onnx_path)
                    split_export_ok = export_split_models(
                        export_model, candidate_onnx_path
                    )
                    if not full_export_ok:
                        raise RuntimeError(
                            "Candidate ONNX export failed due to missing dependencies"
                        )
                    if not split_export_ok:
                        raise RuntimeError(
                            "Candidate split-model export failed due to missing dependencies"
                        )
                    assert_rust_inference_artifacts(candidate_onnx_path)
                    onnx_export_ms = 1000.0 * (time.perf_counter() - onnx_export_start)
                    incumbent_nn_value_weight = generator.incumbent_nn_value_weight()
                    candidate_nn_value_weight = self._compute_candidate_nn_value_weight(
                        current_weight=incumbent_nn_value_weight,
                        config=self.config,
                    )
                    queued = generator.queue_candidate_model(
                        str(candidate_onnx_path),
                        self.step,
                        candidate_nn_value_weight,
                    )
                    logger.info(
                        "Queued candidate model for evaluator",
                        step=self.step,
                        path=str(candidate_onnx_path),
                        queued=queued,
                        onnx_export_ms=onnx_export_ms,
                        incumbent_nn_value_weight=incumbent_nn_value_weight,
                        candidate_nn_value_weight=candidate_nn_value_weight,
                    )
                    if log_to_wandb:
                        wandb.log(
                            {
                                "trainer_step": self.step,
                                "timing/onnx_export_ms": onnx_export_ms,
                                "model_gate/queued_candidate_nn_value_weight": candidate_nn_value_weight,
                            }
                        )
                    next_model_sync_time_s = roll_interval_deadline(
                        next_model_sync_time_s,
                        self.config.run.model_sync_interval_seconds,
                        time.perf_counter(),
                    )

                # Checkpoint (swap to uncompiled model for ONNX export in save)
                if post_step_time >= next_checkpoint_time_s:
                    train_model = self.model
                    self.model = export_model
                    (
                        incumbent_model_artifact,
                        incumbent_model_source_path,
                    ) = self._persist_incumbent_model_artifacts(generator)
                    self.save(
                        extra_checkpoint_state={
                            "incumbent_uses_network": generator.incumbent_uses_network(),
                            "incumbent_model_step": generator.incumbent_model_step(),
                            "incumbent_nn_value_weight": generator.incumbent_nn_value_weight(),
                            "incumbent_eval_avg_attack": generator.incumbent_eval_avg_attack(),
                            "incumbent_model_source_path": incumbent_model_source_path,
                            "incumbent_model_artifact": (
                                incumbent_model_artifact.name
                                if incumbent_model_artifact is not None
                                else None
                            ),
                        }
                    )
                    self.model = train_model
                    next_checkpoint_time_s = roll_interval_deadline(
                        next_checkpoint_time_s,
                        self.config.run.checkpoint_interval_seconds,
                        time.perf_counter(),
                    )

        except KeyboardInterrupt:
            interrupted = True
            logger.info("Training interrupted by user", step=self.step)
        except BaseException as error:
            pending_error = error
            logger.exception("Training loop failed", step=self.step)
        finally:
            # Stop generator
            try:
                logger.info("Stopping game generator")
                generator.stop()
                logger.info(
                    "Game generator stopped",
                    games_generated=generator.games_generated(),
                    examples_generated=generator.examples_generated(),
                )
            except BaseException as error:
                stop_error = error
                logger.exception("Failed to stop game generator cleanly")

            # Restore uncompiled model for final save + ONNX export
            self.model = export_model

            # Always save latest model state on shutdown/interruption.
            (
                incumbent_model_artifact,
                incumbent_model_source_path,
            ) = self._persist_incumbent_model_artifacts(generator)
            final_saved_paths = self.save(
                extra_checkpoint_state={
                    "incumbent_uses_network": generator.incumbent_uses_network(),
                    "incumbent_model_step": generator.incumbent_model_step(),
                    "incumbent_nn_value_weight": generator.incumbent_nn_value_weight(),
                    "incumbent_eval_avg_attack": generator.incumbent_eval_avg_attack(),
                    "incumbent_model_source_path": incumbent_model_source_path,
                    "incumbent_model_artifact": (
                        incumbent_model_artifact.name
                        if incumbent_model_artifact is not None
                        else None
                    ),
                }
            )
            try:
                if log_to_wandb:
                    self._log_final_wandb_model_artifact(final_saved_paths)
                    wandb.finish()
            finally:
                self._cleanup_wandb_gif_files()

        if pending_error is not None:
            raise pending_error
        if stop_error is not None:
            raise stop_error
        if interrupted:
            logger.info("Training stopped cleanly after interrupt", step=self.step)
