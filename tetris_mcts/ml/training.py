"""
Training Loop for Tetris AlphaZero

Implements:
- Training loop with WandB logging
- Learning rate scheduling
- Parallel Rust game generation via GameGenerator
"""

import torch
from typing import Optional
import time
from pathlib import Path
import tempfile
import statistics

import wandb
import structlog

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FPS,
    DEFAULT_GIF_FRAME_DURATION_MS,
    PARALLEL_ONNX_FILENAME,
    TRAINING_DATA_FILENAME,
    TrainingConfig,
)
from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.weights import WeightManager, export_onnx
from tetris_mcts.ml.loss import compute_loss, compute_metrics
from tetris_mcts.ml.evaluation import Evaluator
from tetris_mcts.ml.visualization import create_trajectory_gif

from tetris_core import MCTSConfig, GameGenerator

logger = structlog.get_logger()


class Trainer:
    """Main training class."""

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[TetrisNet] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)

        # Validate paths are set (should be done by setup_run_directory)
        if config.checkpoint_dir is None or config.data_dir is None:
            raise ValueError(
                "checkpoint_dir and data_dir must be set. "
                "Call setup_run_directory() before creating Trainer."
            )

        # Create model
        if model is None:
            model = TetrisNet(
                conv_filters=config.conv_filters,
                fc_hidden=config.fc_hidden,
                conv_kernel_size=config.conv_kernel_size,
                conv_padding=config.conv_padding,
            )
        self.model = model.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create weight manager
        self.weight_manager = WeightManager(config.checkpoint_dir)

        # Create evaluator
        self.evaluator = Evaluator(
            model=self.model,
            checkpoint_dir=config.checkpoint_dir,
            num_simulations=config.num_simulations,
            max_moves=config.max_moves,
            eval_seeds=config.eval_seeds,
            eval_mcts_seed=config.eval_mcts_seed,
        )

        # Training state
        self.step = 0

        # Create directories
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.data_dir.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self):
        if self.config.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_decay_steps,
                eta_min=self.config.learning_rate * self.config.lr_min_factor,
            )
        elif self.config.lr_schedule == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps // self.config.lr_step_divisor,
                gamma=self.config.lr_step_gamma,
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

        if self.config.lr_schedule == "cosine":
            assert isinstance(
                self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            )
            t_max = self.config.lr_decay_steps
            eta_min = self.config.learning_rate * self.config.lr_min_factor
            cosine_factor = (
                1 + torch.cos(torch.tensor(torch.pi * step / t_max))
            ).item() / 2
            lrs = [
                eta_min + (base_lr - eta_min) * cosine_factor
                for base_lr in self.scheduler.base_lrs
            ]
        elif self.config.lr_schedule == "step":
            assert isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR)
            step_size = self.config.lr_decay_steps // self.config.lr_step_divisor
            decay = self.config.lr_step_gamma ** (step // step_size)
            lrs = [base_lr * decay for base_lr in self.scheduler.base_lrs]
        else:
            raise ValueError(f"Unsupported lr_schedule: {self.config.lr_schedule}")

        if len(self.optimizer.param_groups) != len(lrs):
            raise ValueError(
                "Optimizer param group count does not match computed LR count"
            )
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
        self.scheduler._last_lr = lrs

    def train_step(self, batch: tuple) -> dict:
        """Execute one training step."""
        self.model.train()

        boards, aux, policy_targets, value_targets, masks = batch
        boards = boards.to(self.device)
        aux = aux.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        masks = masks.to(self.device)

        # Forward + backward
        self.optimizer.zero_grad()
        total_loss, policy_loss, value_loss = compute_loss(
            self.model, boards, aux, policy_targets, value_targets, masks
        )
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip_norm
        )

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        metrics = {
            "train/loss": total_loss.item(),
            "train/policy_loss": policy_loss.item(),
            "train/value_loss": value_loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch/value_target_mean": value_targets.mean().item(),
            "batch/value_target_std": value_targets.std(unbiased=False).item(),
            "batch/valid_actions_mean": masks.sum(dim=1).mean().item(),
        }

        # Compute additional metrics periodically
        if self.step % self.config.log_interval == 0:
            extra_metrics = compute_metrics(
                self.model, boards, aux, policy_targets, value_targets, masks
            )
            metrics.update(extra_metrics)

        return metrics

    def evaluate(self, render_trajectory: bool = False):
        """Evaluate current model using MCTS on fixed seeds."""
        return self.evaluator.evaluate(render_trajectory)

    def _create_wandb_gif_video(
        self,
        frames: list,
    ) -> tuple[Optional[object], Optional[Path]]:
        if not frames:
            return None, None

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = Path(f.name)

        create_trajectory_gif(
            frames=frames,
            output_path=str(gif_path),
            duration=DEFAULT_GIF_FRAME_DURATION_MS,
        )

        video = wandb.Video(
            str(gif_path),
            fps=DEFAULT_GIF_FPS,
            format="gif",
        )
        return video, gif_path

    def save(self):
        """Save model checkpoint."""
        self.weight_manager.save(
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            export_for_rust=True,
        )
        logger.info("Saved checkpoint", step=self.step)

    def train(self, log_to_wandb: bool = True):
        """
        Run parallel training with Rust game generation in background.

        The Rust GameGenerator runs in a background thread, continuously
        generating games into a shared in-memory buffer. Python samples
        directly from the buffer via generator.sample_batch(). Every
        model_sync_interval steps, a new ONNX model is exported for the
        generator to pick up.

        Args:
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        num_steps = self.config.total_steps
        if self.step >= num_steps:
            logger.info(
                "Target step already reached; skipping training loop",
                target_step=num_steps,
                current_step=self.step,
            )
            return
        model_sync_interval = self.config.model_sync_interval
        # Paths for parallel training (validated in __init__)
        assert self.config.checkpoint_dir is not None
        assert self.config.data_dir is not None
        onnx_path = self.config.checkpoint_dir / PARALLEL_ONNX_FILENAME

        # Export initial model
        export_onnx(self.model, onnx_path)

        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export failed - file not created: {onnx_path}")

        # Create MCTS config for generator
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.config.num_simulations
        mcts_config.c_puct = self.config.c_puct
        mcts_config.temperature = self.config.temperature
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_epsilon = self.config.dirichlet_epsilon
        mcts_config.max_moves = self.config.max_moves

        # Start background game generator
        training_data_path = self.config.data_dir / TRAINING_DATA_FILENAME
        generator = GameGenerator(
            model_path=str(onnx_path),
            training_data_path=str(training_data_path),
            config=mcts_config,
            max_moves=self.config.max_moves,
            add_noise=True,
            max_examples=self.config.buffer_size,
            games_per_save=self.config.games_per_save,
            num_workers=self.config.num_workers,
        )
        generator.start()
        logger.info(
            "Started background game generator",
            model_path=str(onnx_path),
            training_data_path=str(training_data_path),
            num_workers=self.config.num_workers,
        )

        # Wait for minimum buffer size
        logger.info(
            "Waiting for minimum replay buffer size",
            min_examples=self.config.min_buffer_size,
        )
        while generator.buffer_size() < self.config.min_buffer_size:
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

        train_start_time = time.time()

        interrupted = False
        pending_error: BaseException | None = None
        stop_error: BaseException | None = None

        try:
            while self.step < num_steps:
                # Sample batch directly from generator's in-memory buffer
                result = generator.sample_batch(
                    self.config.batch_size,
                    self.config.max_moves,
                )
                if result is None:
                    time.sleep(0.1)
                    continue

                self.step += 1
                session_step = self.step - start_step

                # Convert numpy arrays to torch tensors
                boards, aux, policy_targets, value_targets, masks = result
                batch = (
                    torch.from_numpy(boards).reshape(
                        -1, 1, BOARD_HEIGHT, BOARD_WIDTH
                    ),  # [batch, 1, H, W]
                    torch.from_numpy(aux),
                    torch.from_numpy(policy_targets),
                    torch.from_numpy(value_targets),
                    torch.from_numpy(masks),
                )

                # Train step
                metrics = self.train_step(batch)

                # Log metrics
                if self.step % self.config.log_interval == 0:
                    elapsed = time.time() - train_start_time
                    games = generator.games_generated()
                    metrics["buffer_size"] = generator.buffer_size()
                    metrics["games_generated"] = games
                    metrics["examples_generated"] = generator.examples_generated()
                    metrics["games_per_second"] = games / elapsed if elapsed > 0 else 0
                    metrics["steps_per_second"] = (
                        session_step / elapsed if elapsed > 0 else 0
                    )
                    if log_to_wandb:
                        metrics["trainer_step"] = self.step
                        wandb.log(metrics)
                        # Log all completed games since the last logging tick.
                        for (
                            game_number,
                            game_stats,
                        ) in generator.drain_completed_game_stats():
                            game_metrics = {
                                "game_number": game_number,
                                "trainer_step": self.step,
                            }
                            for key, value in game_stats.items():
                                game_metrics[f"game/{key}"] = value
                            episode_length = game_stats["episode_length"]
                            if episode_length <= 0:
                                raise ValueError(
                                    f"Invalid episode_length for game {game_number}: "
                                    f"{episode_length}"
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
                            # Don't pin per-game logs to the training step: multiple
                            # games can complete between train ticks, and reusing the
                            # same step causes only a subset to appear in history.
                            wandb.log(game_metrics)
                    logger.info(
                        "Training progress",
                        step=self.step,
                        loss=metrics["train/loss"],
                        learning_rate=metrics["train/learning_rate"],
                        buffer_size=generator.buffer_size(),
                        games_generated=games,
                        games_per_second=metrics["games_per_second"],
                        steps_per_second=metrics["steps_per_second"],
                    )

                # Export updated model for generator
                if self.step % model_sync_interval == 0:
                    export_onnx(self.model, onnx_path)
                    logger.info(
                        "Exported model for generator",
                        step=self.step,
                        path=str(onnx_path),
                    )

                # Evaluate
                if self.step % self.config.eval_interval == 0:
                    # Render trajectory every evaluation for visualization
                    eval_result, trajectory_frames = self.evaluate(
                        render_trajectory=log_to_wandb
                    )
                    if log_to_wandb:
                        eval_gif_path: Optional[Path] = None
                        attacks = [attack for attack, _ in eval_result.game_results]
                        moves = [moves for _, moves in eval_result.game_results]
                        log_data = {
                            "eval/num_games": eval_result.num_games,
                            "eval/avg_attack": eval_result.avg_attack,
                            "eval/max_attack": eval_result.max_attack,
                            "eval/avg_lines": eval_result.avg_lines,
                            "eval/max_lines": eval_result.max_lines,
                            "eval/avg_moves": eval_result.avg_moves,
                            "eval/attack_per_piece": eval_result.attack_per_piece,
                            "eval/lines_per_piece": eval_result.lines_per_piece,
                            "eval/attack_std": statistics.pstdev(attacks),
                            "eval/moves_std": statistics.pstdev(moves),
                            "trainer_step": self.step,
                        }
                        # Log trajectory as animated GIF
                        if trajectory_frames:
                            eval_video, eval_gif_path = self._create_wandb_gif_video(
                                trajectory_frames
                            )
                            if eval_video is not None:
                                log_data["eval/trajectory"] = eval_video
                        wandb.log(log_data)

                # Checkpoint
                if self.step % self.config.checkpoint_interval == 0:
                    self.save()

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

            # Always save latest model state on shutdown/interruption.
            self.save()
            if log_to_wandb:
                wandb.finish()

        if pending_error is not None:
            raise pending_error
        if stop_error is not None:
            raise stop_error
        if interrupted:
            logger.info("Training stopped cleanly after interrupt", step=self.step)
