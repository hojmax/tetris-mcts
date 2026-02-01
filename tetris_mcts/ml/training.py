"""
Training Loop for Tetris AlphaZero

Implements:
- Training loop with WandB logging
- Learning rate scheduling
- Parallel Rust game generation via GameGenerator
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time

import wandb

from tetris_mcts.ml.network import TetrisNet, MAX_MOVES
from tetris_mcts.ml.weights import WeightManager, export_onnx
from tetris_mcts.ml.loss import compute_loss, compute_metrics
from tetris_mcts.ml.evaluation import Evaluator

from tetris_core import MCTSConfig, GameGenerator


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Network
    conv_filters: list[int] = field(default_factory=lambda: [4, 8])
    fc_hidden: int = 128

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"  # 'cosine', 'step', 'none'
    lr_decay_steps: int = 100000

    # Self-play
    num_simulations: int = 100  # MCTS simulations per move
    temperature: float = 1.0
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25

    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 10_000
    games_per_save: int = 100  # Games between disk saves (0 to disable)

    # Iteration
    training_steps_per_iter: int = 1000
    checkpoint_interval: int = 10
    eval_interval: int = 100
    eval_seeds: list[int] = field(default_factory=lambda: list(range(20)))

    # Paths (relative to project root)
    checkpoint_dir: str = "outputs/checkpoints"
    data_dir: str = "outputs/data"

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: Optional[str] = None
    log_interval: int = 100


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

        # Create model
        if model is None:
            model = TetrisNet(
                conv_filters=config.conv_filters,
                fc_hidden=config.fc_hidden,
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
            eval_seeds=config.eval_seeds,
        )

        # Training state
        self.step = 0
        self.iteration = 0

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self):
        if self.config.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_decay_steps,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.lr_schedule == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps // 3,
                gamma=0.1,
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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

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

    def save(self):
        """Save model checkpoint."""
        self.weight_manager.save(
            self.model,
            self.optimizer,
            self.step,
            export_for_rust=True,
        )
        print(f"Saved checkpoint at step {self.step}")

    def load(self) -> bool:
        """Load latest checkpoint if available."""
        step = self.weight_manager.load_latest(self.model, self.optimizer)
        if step is not None:
            self.step = step
            print(f"Loaded checkpoint at step {step}")
            return True
        return False

    def train(
        self,
        num_steps: int = 100000,
        model_sync_interval: int = 1000,
        log_to_wandb: bool = True,
    ):
        """
        Run parallel training with Rust game generation in background.

        The Rust GameGenerator runs in a background thread, continuously
        generating games into a shared in-memory buffer. Python samples
        directly from the buffer via generator.sample_batch(). Every
        model_sync_interval steps, a new ONNX model is exported for the
        generator to pick up.

        Args:
            num_steps: Total number of training steps
            model_sync_interval: Steps between model exports
            log_to_wandb: Whether to log metrics to Weights & Biases
        """
        # Initialize wandb
        if log_to_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
            )

        # Paths for parallel training
        games_dir = Path(self.config.data_dir) / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = Path(self.config.checkpoint_dir) / "parallel.onnx"

        # Export initial model
        export_onnx(self.model, onnx_path)

        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export failed - file not created: {onnx_path}")

        # Create MCTS config for generator
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.config.num_simulations
        mcts_config.temperature = self.config.temperature
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_epsilon = self.config.dirichlet_epsilon

        # Start background game generator
        generator = GameGenerator(
            model_path=str(onnx_path),
            output_dir=str(games_dir),
            config=mcts_config,
            max_moves=MAX_MOVES,
            add_noise=True,
            max_examples=self.config.buffer_size,
            games_per_save=self.config.games_per_save,
        )
        generator.start()
        print("Started background game generator")
        print(f"  Model path: {onnx_path}")
        print(f"  Output dir: {games_dir}")

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
                    torch.from_numpy(boards).unsqueeze(1),  # Add channel dim
                    torch.from_numpy(aux),
                    torch.from_numpy(policy_targets),
                    torch.from_numpy(value_targets),
                    torch.from_numpy(masks),
                )

                # Train step
                metrics = self.train_step(batch)

                # Log metrics
                if self.step % self.config.log_interval == 0:
                    metrics["buffer_size"] = generator.buffer_size()
                    metrics["games_generated"] = generator.games_generated()
                    metrics["examples_generated"] = generator.examples_generated()
                    if log_to_wandb:
                        wandb.log(metrics, step=self.step)
                    print(
                        f"Step {self.step}: loss={metrics['loss']:.4f}, "
                        f"buffer={generator.buffer_size()}, "
                        f"games={generator.games_generated()}"
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
                        log_data = {
                            "eval/avg_attack": eval_result.avg_attack,
                            "eval/max_attack": eval_result.max_attack,
                            "eval/avg_moves": eval_result.avg_moves,
                            "eval/attack_per_piece": eval_result.attack_per_piece,
                        }
                        # Log trajectory as animated GIF
                        if trajectory_frames:
                            import tempfile

                            with tempfile.NamedTemporaryFile(
                                suffix=".gif", delete=False
                            ) as f:
                                gif_path = f.name
                            trajectory_frames[0].save(
                                gif_path,
                                save_all=True,
                                append_images=trajectory_frames[1:],
                                duration=300,  # ms per frame
                                loop=0,
                            )
                            log_data["eval/trajectory"] = wandb.Video(
                                gif_path, fps=3, format="gif"
                            )
                        wandb.log(log_data, step=self.step)

                # Checkpoint
                if (
                    self.step
                    % (
                        self.config.checkpoint_interval
                        * self.config.training_steps_per_iter
                    )
                    == 0
                ):
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
