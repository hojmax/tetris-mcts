"""
Training Loop for Tetris AlphaZero

Implements:
- Loss computation (policy CE + value MSE)
- Training loop with WandB logging
- Learning rate scheduling
- Evaluation during training
- Parallel Rust game generation via GameGenerator
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time

import wandb

from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.data import ReplayBuffer, SharedReplayBuffer, TetrisDataset, TrainingExample
from tetris_mcts.ml.weights import WeightManager

from tetris_core import MCTSConfig, MCTSAgent, GameGenerator, evaluate_model, EvalResult


# Game configuration
MAX_MOVES = 100


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
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000

    # Self-play
    num_simulations: int = 100  # MCTS simulations per move
    temperature: float = 1.0
    temperature_drop_move: int = 15
    temperature_final: float = 0.1
    num_games_per_iteration: int = 100
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25

    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 10_000

    # Iteration
    num_iterations: int = 100
    training_steps_per_iter: int = 1000
    checkpoint_interval: int = 10
    eval_interval: int = 100
    eval_seeds: list[int] = field(default_factory=lambda: list(range(20)))

    # Paths
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: Optional[str] = None
    log_interval: int = 100


def compute_loss(
    model: TetrisNet,
    boards: torch.Tensor,
    aux_features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    action_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined policy and value loss.

    Args:
        model: TetrisNet model
        boards: (batch, 1, 20, 10)
        aux_features: (batch, 52)
        policy_targets: (batch, 734) - MCTS policy targets
        value_targets: (batch,) - discounted attack targets
        action_masks: (batch, 734) - valid action masks

    Returns:
        total_loss, policy_loss, value_loss
    """
    # Forward pass
    policy_logits, value_pred = model(boards, aux_features)

    # Ensure at least one valid action per sample (avoid all -inf)
    # This handles edge cases where mask might be all zeros
    valid_counts = action_masks.sum(dim=1, keepdim=True)
    safe_masks = action_masks.clone()
    safe_masks[valid_counts.squeeze() == 0, 0] = 1  # Set first action as valid if none

    # Apply mask before softmax
    masked_logits = policy_logits.masked_fill(safe_masks == 0, float("-inf"))
    log_policy = F.log_softmax(masked_logits, dim=-1)

    # Replace -inf with 0 for numerical stability (won't contribute to loss anyway
    # since policy_targets should be 0 for invalid actions)
    log_policy = torch.where(
        torch.isinf(log_policy), torch.zeros_like(log_policy), log_policy
    )

    # Policy loss: cross-entropy with MCTS policy
    # -sum(target * log(pred))
    policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)

    # Total loss (AlphaZero uses equal weighting)
    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss


def compute_metrics(
    model: TetrisNet,
    boards: torch.Tensor,
    aux_features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    action_masks: torch.Tensor,
) -> dict:
    """Compute additional metrics for logging."""
    with torch.no_grad():
        policy_logits, value_pred = model(boards, aux_features)

        # Ensure at least one valid action per sample (avoid all -inf)
        valid_counts = action_masks.sum(dim=1, keepdim=True)
        safe_masks = action_masks.clone()
        safe_masks[valid_counts.squeeze() == 0, 0] = 1

        # Apply mask
        masked_logits = policy_logits.masked_fill(safe_masks == 0, float("-inf"))
        policy_probs = F.softmax(masked_logits, dim=-1)

        # Policy entropy
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -torch.sum(policy_probs * log_probs, dim=-1).mean()

        # Value prediction error
        value_error = torch.abs(value_pred.squeeze(-1) - value_targets).mean()

        # Top-1 accuracy (if target is argmax of MCTS policy)
        pred_actions = policy_probs.argmax(dim=-1)
        target_actions = policy_targets.argmax(dim=-1)
        top1_acc = (pred_actions == target_actions).float().mean()

    return {
        "policy_entropy": entropy.item(),
        "value_error": value_error.item(),
        "top1_accuracy": top1_acc.item(),
    }


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

        # Create replay buffer
        self.buffer = ReplayBuffer(max_size=config.buffer_size)

        # Create weight manager
        self.weight_manager = WeightManager(config.checkpoint_dir)

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

    def generate_data(self, num_games: int) -> list[TrainingExample]:
        """Generate self-play data using MCTS."""
        print(f"Generating {num_games} games...")

        # Export current model to ONNX for Rust inference
        onnx_path = Path(self.config.checkpoint_dir) / "selfplay.onnx"
        from .weights import export_onnx
        export_onnx(self.model, onnx_path)

        # Configure MCTS
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.config.num_simulations
        mcts_config.temperature = self.config.temperature
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_epsilon = self.config.dirichlet_epsilon

        # Create agent and load model
        agent = MCTSAgent(mcts_config)
        if not agent.load_model(str(onnx_path)):
            raise RuntimeError(f"Failed to load model from {onnx_path}")

        start_time = time.time()

        # Generate games in Rust
        rust_examples = agent.generate_games(
            num_games=num_games,
            max_moves=MAX_MOVES,
            add_noise=True,
            drop_last_n=10,
        )

        # Convert Rust examples to Python TrainingExample objects
        examples = [self._rust_example_to_training_example(ex) for ex in rust_examples]

        elapsed = time.time() - start_time
        print(f"Generated {len(examples)} examples from {num_games} games")
        print(f"Time: {elapsed:.1f}s ({num_games / max(0.1, elapsed):.2f} games/sec)")

        return examples

    def _rust_example_to_training_example(self, rust_ex) -> TrainingExample:
        """Convert Rust TrainingExample to Python TrainingExample."""
        import numpy as np
        BOARD_HEIGHT, BOARD_WIDTH = 20, 10

        # Reshape board from flat list to 2D array
        board = np.array(rust_ex.board, dtype=np.uint8).reshape(BOARD_HEIGHT, BOARD_WIDTH)

        return TrainingExample(
            board=board.astype(bool),
            current_piece=rust_ex.current_piece,
            hold_piece=rust_ex.hold_piece if rust_ex.hold_piece < 7 else None,
            hold_available=rust_ex.hold_available,
            next_queue=list(rust_ex.next_queue),
            move_number=rust_ex.move_number,
            policy_target=np.array(rust_ex.policy, dtype=np.float32),
            value_target=rust_ex.value,
            action_mask=np.array(rust_ex.action_mask, dtype=bool),
        )

    def evaluate(self) -> EvalResult:
        """Evaluate current model using MCTS on fixed seeds."""
        self.model.eval()

        # Export model to ONNX for Rust evaluation
        onnx_path = Path(self.config.checkpoint_dir) / "eval.onnx"
        from .weights import export_onnx
        export_onnx(self.model, onnx_path)

        # Create MCTS config for evaluation (temperature=0 enforced by evaluate_model)
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = self.config.num_simulations

        # Run evaluation in Rust with seeded environments
        result = evaluate_model(
            model_path=str(onnx_path),
            seeds=[int(s) for s in self.config.eval_seeds],
            config=mcts_config,
            max_moves=MAX_MOVES,
        )

        print(f"Evaluation ({result.num_games} games):")
        print(f"  Avg attack: {result.avg_attack:.1f}")
        print(f"  Max attack: {result.max_attack}")
        print(f"  Avg moves: {result.avg_moves:.1f}")
        print(f"  Attack/piece: {result.attack_per_piece:.3f}")

        return result

    def train_iteration(self, log_to_wandb: bool = True) -> dict:
        """Run one training iteration."""
        self.iteration += 1
        iteration_metrics = {}

        # Generate self-play data
        start_time = time.time()
        examples = self.generate_data(self.config.num_games_per_iteration)
        self.buffer.add_batch(examples)
        data_time = time.time() - start_time

        iteration_metrics["data_generation_time"] = data_time
        iteration_metrics["buffer_size"] = len(self.buffer)
        iteration_metrics["new_examples"] = len(examples)

        # Training steps
        if len(self.buffer) >= self.config.min_buffer_size:
            train_start = time.time()
            step_metrics = []

            for _ in range(self.config.training_steps_per_iter):
                batch = self.buffer.sample(self.config.batch_size)
                metrics = self.train_step(batch)
                step_metrics.append(metrics)

                # Log to wandb
                if log_to_wandb and self.step % self.config.log_interval == 0:
                    wandb.log(metrics, step=self.step)

            train_time = time.time() - train_start

            # Average metrics
            avg_metrics = {}
            for key in step_metrics[0]:
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in step_metrics])
            iteration_metrics.update(avg_metrics)
            iteration_metrics["training_time"] = train_time

        # Evaluate
        if self.iteration % self.config.eval_interval == 0:
            eval_result = self.evaluate()
            iteration_metrics["eval_avg_attack"] = eval_result.avg_attack
            iteration_metrics["eval_max_attack"] = eval_result.max_attack
            iteration_metrics["eval_avg_moves"] = eval_result.avg_moves
            iteration_metrics["eval_attack_per_piece"] = eval_result.attack_per_piece

            if log_to_wandb:
                wandb.log({
                    "eval/avg_attack": eval_result.avg_attack,
                    "eval/max_attack": eval_result.max_attack,
                    "eval/avg_moves": eval_result.avg_moves,
                    "eval/attack_per_piece": eval_result.attack_per_piece,
                }, step=self.step)

        # Save checkpoint
        if self.iteration % self.config.checkpoint_interval == 0:
            self.save()

        return iteration_metrics

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

    def train(self, num_iterations: Optional[int] = None):
        """Run full training loop."""
        if num_iterations is None:
            num_iterations = self.config.num_iterations

        # Initialize wandb
        wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            config=vars(self.config),
        )

        print(f"Starting training for {num_iterations} iterations")
        print(f"Config: {self.config}")
        print()

        for i in range(num_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {i + 1}/{num_iterations}")
            print(f"{'=' * 60}")

            metrics = self.train_iteration()

            print(f"Buffer size: {metrics['buffer_size']}")
            if "avg_loss" in metrics:
                print(f"Avg loss: {metrics['avg_loss']:.4f}")
                print(f"Avg policy loss: {metrics['avg_policy_loss']:.4f}")
                print(f"Avg value loss: {metrics['avg_value_loss']:.4f}")

        # Final save
        self.save()

        wandb.finish()

    def train_parallel(
        self,
        num_steps: int = 100000,
        model_sync_interval: int = 1000,
    ):
        """
        Run parallel training with Rust game generation in background.

        The Rust GameGenerator runs in a background thread, continuously
        generating games and writing them to disk. Python reads from disk
        via SharedReplayBuffer and trains the model. Every model_sync_interval
        steps, a new ONNX model is exported for the generator to pick up.

        Args:
            num_steps: Total number of training steps
            model_sync_interval: Steps between model exports
        """
        # Initialize wandb
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
        from .weights import export_onnx
        export_onnx(self.model, onnx_path)

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
        )
        generator.start()
        print(f"Started background game generator")
        print(f"  Model path: {onnx_path}")
        print(f"  Output dir: {games_dir}")

        # Create shared replay buffer
        shared_buffer = SharedReplayBuffer(games_dir, max_files=100)

        # Wait for minimum buffer size
        print(f"Waiting for {self.config.min_buffer_size} examples...")
        while shared_buffer.size() < self.config.min_buffer_size:
            time.sleep(1.0)
            print(f"  Buffer: {shared_buffer.size()} examples, "
                  f"Games: {generator.games_generated()}")

        print(f"\nStarting training for {num_steps} steps")
        print(f"Config: {self.config}")

        try:
            for step in range(num_steps):
                self.step = step + 1

                # Sample batch from shared buffer
                batch = shared_buffer.sample(self.config.batch_size)
                if batch is None:
                    time.sleep(0.1)
                    continue

                # Train step
                metrics = self.train_step(batch)

                # Log to wandb
                if self.step % self.config.log_interval == 0:
                    metrics["buffer_size"] = shared_buffer.size()
                    metrics["games_generated"] = generator.games_generated()
                    metrics["examples_generated"] = generator.examples_generated()
                    wandb.log(metrics, step=self.step)
                    print(f"Step {self.step}: loss={metrics['loss']:.4f}, "
                          f"buffer={shared_buffer.size()}, "
                          f"games={generator.games_generated()}")

                # Export updated model for generator
                if self.step % model_sync_interval == 0:
                    export_onnx(self.model, onnx_path)
                    print(f"  Exported model at step {self.step}")

                # Evaluate
                if self.step % self.config.eval_interval == 0:
                    eval_result = self.evaluate()
                    wandb.log({
                        "eval/avg_attack": eval_result.avg_attack,
                        "eval/max_attack": eval_result.max_attack,
                        "eval/avg_moves": eval_result.avg_moves,
                        "eval/attack_per_piece": eval_result.attack_per_piece,
                    }, step=self.step)

                # Checkpoint
                if self.step % (self.config.checkpoint_interval * self.config.training_steps_per_iter) == 0:
                    self.save()

        finally:
            # Stop generator
            print("\nStopping game generator...")
            generator.stop()
            print(f"Final stats: {generator.games_generated()} games, "
                  f"{generator.examples_generated()} examples")

        # Final save
        self.save()
        wandb.finish()


def train_from_data(
    data_path: str | Path,
    config: Optional[TrainingConfig] = None,
    num_epochs: int = 10,
    device: str = "cpu",
):
    """Train from pre-generated data file."""
    if config is None:
        config = TrainingConfig()

    # Load data
    dataset = TetrisDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Create model and trainer
    model = TetrisNet(
        conv_filters=config.conv_filters,
        fc_hidden=config.fc_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    print(f"Training on {len(dataset)} examples for {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []

        for batch in dataloader:
            boards, aux, policy_targets, value_targets, masks = batch
            boards = boards.to(device)
            aux = aux.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            total_loss, policy_loss, value_loss = compute_loss(
                model, boards, aux, policy_targets, value_targets, masks
            )
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"loss={np.mean(epoch_losses):.4f}, "
            f"policy={np.mean(epoch_policy_losses):.4f}, "
            f"value={np.mean(epoch_value_losses):.4f}"
        )

    return model


if __name__ == "__main__":
    import tempfile

    print("Testing training module...")
    print()

    # Create config with minimal settings for testing
    config = TrainingConfig(
        batch_size=32,
        num_games_per_iteration=10,
        training_steps_per_iter=50,
        min_buffer_size=100,
        checkpoint_interval=1,
        eval_interval=1,
        log_interval=10,
    )

    # Use temp directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = tmpdir
        config.data_dir = tmpdir

        # Create trainer
        trainer = Trainer(config, device="cpu")

        print("Model parameters:", sum(p.numel() for p in trainer.model.parameters()))
        print()

        # Run one iteration
        print("Running single training iteration...")
        metrics = trainer.train_iteration(log_to_wandb=False)
        print(f"Metrics: {metrics}")
        print()

        # Test save/load
        print("Testing save/load...")
        trainer.save()
        trainer2 = Trainer(config, device="cpu")
        loaded = trainer2.load()
        print(f"Loaded: {loaded}, step: {trainer2.step}")

    print("\nAll tests passed!")
