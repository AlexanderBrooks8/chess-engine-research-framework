from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minizero.models.transformer_policy_value import TransformerPolicyValue
from minizero.selfplay.mcts_worker import generate_mcts_selfplay_games
from minizero.train.dataset import (
    ReplayDataset,
    collate_training_examples,
    split_batch,
)
from minizero.train.losses import combined_policy_value_loss
from minizero.utils.checkpoint import save_checkpoint
from minizero.models.model_specs import MODEL_CONFIG


def default_model_config() -> dict:
    return MODEL_CONFIG


def load_model_and_config(
    checkpoint_path: str | Path | None,
) -> tuple[TransformerPolicyValue, dict]:
    if checkpoint_path is None:
        model_config = default_model_config()
        model = TransformerPolicyValue(**model_config)
        return model, model_config

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )

    model_config = checkpoint.get("model_config", default_model_config())
    model = TransformerPolicyValue(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, model_config


def save_initial_checkpoint(
    checkpoint_path: Path,
    model_config: dict,
) -> None:
    model = TransformerPolicyValue(**model_config)

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        step=0,
        extra={"description": "initial random MCTS self-play checkpoint"},
        model_config=model_config,
    )


def train_for_steps(
    checkpoint_in: Path,
    checkpoint_out: Path,
    replay_paths: list[Path],
    batch_size: int,
    train_steps: int,
    lr: float,
    value_weight: float,
) -> dict[str, float]:
    model, model_config = load_model_and_config(checkpoint_in)

    dataset = ReplayDataset(replay_paths)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_training_examples,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    last_total_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0

    data_iter = iter(dataloader)

    for _ in range(train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        model_inputs, target_policy, target_value, legal_mask = split_batch(batch)

        output = model(**model_inputs)

        total_loss, policy_loss, value_loss = combined_policy_value_loss(
            policy_logits=output.policy_logits,
            target_policy=target_policy,
            predicted_value=output.value,
            target_value=target_value,
            value_weight=value_weight,
            legal_mask=legal_mask,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_total_loss = float(total_loss.item())
        last_policy_loss = float(policy_loss.item())
        last_value_loss = float(value_loss.item())

    save_checkpoint(
        path=checkpoint_out,
        model=model,
        optimizer=optimizer,
        step=train_steps,
        extra={
            "description": "iterative MCTS self-play checkpoint",
            "num_replay_files": len(replay_paths),
            "num_examples": len(dataset),
            "train_steps": train_steps,
            "batch_size": batch_size,
        },
        model_config=model_config,
    )

    return {
        "loss": last_total_loss,
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "num_examples": float(len(dataset)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative MCTS self-play training.")

    parser.add_argument("--run-dir", type=str, default="runs/iterative_mcts_v1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/iterative_mcts_v1")
    parser.add_argument("--checkpoint-in", type=str, default=None)

    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games-per-iter", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=30)
    parser.add_argument("--num-simulations", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)

    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Train only on current iteration records instead of accumulated replay.",
    )
    
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--disable-root-noise", action="store_true")
    
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(1337)

    run_dir = Path(args.run_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    replay_paths: list[Path] = []

    if args.checkpoint_in is None:
        current_checkpoint = checkpoint_dir / "iter_0000.pt"
        save_initial_checkpoint(
            checkpoint_path=current_checkpoint,
            model_config=default_model_config(),
        )
    else:
        current_checkpoint = Path(args.checkpoint_in)

    print()
    print("Iterative MCTS self-play")
    print("-" * 40)
    print(f"Run dir:          {run_dir}")
    print(f"Checkpoint dir:   {checkpoint_dir}")
    print(f"Start checkpoint: {current_checkpoint}")
    print(f"Iterations:       {args.iterations}")
    print(f"Games/iter:       {args.games_per_iter}")
    print(f"Sims/move:        {args.num_simulations}")
    print(f"Train steps/iter: {args.train_steps}")
    print(f"Workers:          {args.num_workers}")
    print()
    
    root_dirichlet_alpha = None if args.disable_root_noise else args.root_dirichlet_alpha

    for iteration in range(1, args.iterations + 1):
        iter_dir = run_dir / f"iter_{iteration:04d}"
        selfplay_dir = iter_dir / "selfplay"

        saved_paths = generate_mcts_selfplay_games(
            output_dir=selfplay_dir,
            checkpoint_path=current_checkpoint,
            games=args.games_per_iter,
            max_plies=args.max_plies,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            num_workers=args.num_workers,
        )

        if args.latest_only:
            train_paths = saved_paths
        else:
            replay_paths.extend(saved_paths)
            train_paths = replay_paths

        next_checkpoint = checkpoint_dir / f"iter_{iteration:04d}.pt"

        stats = train_for_steps(
            checkpoint_in=current_checkpoint,
            checkpoint_out=next_checkpoint,
            replay_paths=train_paths,
            batch_size=args.batch_size,
            train_steps=args.train_steps,
            lr=args.lr,
            value_weight=args.value_weight,
        )

        print(f"Iteration {iteration:04d}")
        print("-" * 40)
        print(f"Generated games:   {len(saved_paths)}")
        print(f"Replay files used: {len(train_paths)}")
        print(f"Examples used:     {int(stats['num_examples'])}")
        print(f"Loss:              {stats['loss']:.6f}")
        print(f"Policy loss:       {stats['policy_loss']:.6f}")
        print(f"Value loss:        {stats['value_loss']:.6f}")
        print(f"Saved checkpoint:  {next_checkpoint}")
        print()

        current_checkpoint = next_checkpoint

    print("Done.")
    print(f"Final checkpoint: {current_checkpoint}")
    print()


if __name__ == "__main__":
    main()