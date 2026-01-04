#!/usr/bin/env python3
"""
Tetris AlphaZero Training Script

Usage:
    python train.py --iterations 100 --games-per-iter 50 --simulations 100

For a quick test:
    python train.py --iterations 5 --games-per-iter 10 --simulations 20
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.training import Trainer, TrainingConfig
from ml.network import TetrisNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tetris AlphaZero')

    # Training settings
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=50,
                        help='Games to play per iteration')
    parser.add_argument('--train-steps-per-iter', type=int, default=500,
                        help='Training steps per iteration')

    # MCTS settings
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for action selection')
    parser.add_argument('--no-mcts', action='store_true',
                        help='Use random policy instead of MCTS (for debugging)')

    # Network settings
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    # Buffer settings
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--min-buffer', type=int, default=1000,
                        help='Minimum buffer size before training')

    # Logging/checkpoints
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate every N iterations')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log every N training steps')

    # WandB
    parser.add_argument('--project', type=str, default='tetris-alphazero',
                        help='WandB project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='WandB run name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda/mps)')

    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Tetris AlphaZero Training")
    print("="*60)
    print()

    # Create config
    config = TrainingConfig(
        # Training
        batch_size=args.batch_size,
        learning_rate=args.lr,

        # Self-play
        num_simulations=args.simulations,
        temperature=args.temperature,
        num_games_per_iteration=args.games_per_iter,

        # Buffer
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer,

        # Iterations
        num_iterations=args.iterations,
        training_steps_per_iter=args.train_steps_per_iter,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,

        # Paths
        checkpoint_dir=args.checkpoint_dir,

        # WandB
        project_name=args.project,
        run_name=args.run_name,
    )

    print(f"Configuration:")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Games per iteration: {config.num_games_per_iteration}")
    print(f"  MCTS simulations: {config.num_simulations}")
    print(f"  Training steps per iteration: {config.training_steps_per_iter}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Buffer size: {config.buffer_size}")
    print(f"  Device: {args.device}")
    print(f"  Use MCTS: {not args.no_mcts}")
    print()

    # Create trainer
    trainer = Trainer(config, device=args.device)

    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print()

    # Resume if requested
    if args.resume:
        if trainer.load():
            print(f"Resumed from step {trainer.step}")
        else:
            print("No checkpoint found, starting fresh")
        print()

    # Run training
    use_mcts = not args.no_mcts
    log_to_wandb = not args.no_wandb

    print("Starting training...")
    print()

    for i in range(config.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i + 1}/{config.num_iterations}")
        print(f"{'='*60}")

        metrics = trainer.train_iteration(log_to_wandb=log_to_wandb, use_mcts=use_mcts)

        print(f"Buffer size: {metrics['buffer_size']}")
        if 'avg_loss' in metrics:
            print(f"Avg loss: {metrics['avg_loss']:.4f}")
            print(f"Avg policy loss: {metrics['avg_policy_loss']:.4f}")
            print(f"Avg value loss: {metrics['avg_value_loss']:.4f}")

    # Final save
    trainer.save()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
