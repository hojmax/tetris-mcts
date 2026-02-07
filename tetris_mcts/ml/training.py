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

import wandb

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_GIF_FPS,
    DEFAULT_GIF_FRAME_DURATION_MS,
    PARALLEL_ONNX_FILENAME,
    TrainingConfig,
)
from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.weights import WeightManager, export_onnx
from tetris_mcts.ml.loss import compute_loss, compute_metrics
from tetris_mcts.ml.evaluation import Evaluator
from tetris_mcts.ml.visualization import create_trajectory_gif

from tetris_core import MCTSConfig, GameGenerator


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

        self.step += 1

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
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
        print(f"Saved checkpoint at step {self.step}")

    def load(self) -> bool:
        """Load latest checkpoint if available."""
        step = self.weight_manager.load_latest(
            self.model,
            self.optimizer,
            self.scheduler,
        )
        if step is not None:
            self.step = step
            print(f"Loaded checkpoint at step {step}")
            return True
        return False

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

        # Start background game generator
        # data_dir is the run directory, training_data.npz will be saved there
        generator = GameGenerator(
            model_path=str(onnx_path),
            output_dir=str(self.config.data_dir),
            config=mcts_config,
            max_moves=self.config.max_moves,
            add_noise=True,
            max_examples=self.config.buffer_size,
            games_per_save=self.config.games_per_save,
            num_workers=self.config.num_workers,
        )
        generator.start()
        print("Started background game generator")
        print(f"  Model path: {onnx_path}")
        print(f"  Data dir: {self.config.data_dir}")

        # Wait for minimum buffer size
        print(f"Waiting for {self.config.min_buffer_size} examples...")
        while generator.buffer_size() < self.config.min_buffer_size:
            time.sleep(1.0)
            print(
                f"  Buffer: {generator.buffer_size()} examples, "
                f"Games: {generator.games_generated()}"
            )

        print(f"\nStarting training for {num_steps} steps")
        print(f"Config: {self.config}")

        last_logged_game = 0  # Track which game we last logged
        train_start_time = time.time()

        try:
            for step in range(num_steps):
                self.step = step + 1

                # Sample batch directly from generator's in-memory buffer
                result = generator.sample_batch(self.config.batch_size)
                if result is None:
                    time.sleep(0.1)
                    continue

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
                        self.step / elapsed if elapsed > 0 else 0
                    )
                    if log_to_wandb:
                        wandb.log(metrics, step=self.step)
                        # Log per-game stats with game_number as x-axis
                        last_game = generator.get_last_game_stats()
                        if last_game is not None:
                            game_number, game_stats = last_game
                            if game_number != last_logged_game:
                                game_metrics = {"game_number": game_number}
                                for key, value in game_stats.items():
                                    game_metrics[f"game/{key}"] = value
                                wandb.log(game_metrics)
                                last_logged_game = game_number
                    print(
                        f"Step {self.step}: loss={metrics['loss']:.4f}, "
                        f"lr={metrics['learning_rate']:.2e}, "
                        f"buffer={generator.buffer_size()}, "
                        f"games={games}, "
                        f"games/s={metrics['games_per_second']:.2f}, "
                        f"steps/s={metrics['steps_per_second']:.1f}"
                    )

                # Export updated model for generator
                if self.step % model_sync_interval == 0:
                    export_onnx(self.model, onnx_path)
                    print(f"  Exported model at step {self.step}")

                # Evaluate
                if self.step % self.config.eval_interval == 0:
                    # Render trajectory every evaluation for visualization
                    eval_result, trajectory_frames = self.evaluate(
                        render_trajectory=log_to_wandb
                    )
                    if log_to_wandb:
                        eval_gif_path: Optional[Path] = None
                        log_data = {
                            "eval/avg_attack": eval_result.avg_attack,
                            "eval/max_attack": eval_result.max_attack,
                            "eval/avg_moves": eval_result.avg_moves,
                            "eval/attack_per_piece": eval_result.attack_per_piece,
                        }
                        # Log trajectory as animated GIF
                        if trajectory_frames:
                            eval_video, eval_gif_path = self._create_wandb_gif_video(
                                trajectory_frames
                            )
                            if eval_video is not None:
                                log_data["eval/trajectory"] = eval_video
                        wandb.log(log_data, step=self.step)
                        if eval_gif_path is not None:
                            eval_gif_path.unlink(missing_ok=True)

                # Checkpoint
                if self.step % self.config.checkpoint_interval == 0:
                    self.save()

        finally:
            # Stop generator
            print("\nStopping game generator...")
            generator.stop()
            print(
                f"Final stats: {generator.games_generated()} games, "
                f"{generator.examples_generated()} examples"
            )

        # Final save
        self.save()
        if log_to_wandb:
            wandb.finish()
