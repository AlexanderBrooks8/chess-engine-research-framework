from __future__ import annotations

import argparse
from pathlib import Path

from minizero.selfplay.mcts_worker import generate_mcts_selfplay_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MCTS-guided self-play records.")

    parser.add_argument("--output-dir", type=str, default="runs/mcts_selfplay")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--num-simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument(
        "--disable-root-noise",
        action="store_true",
        help="Disable Dirichlet root noise during MCTS self-play.",
    )
    
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    root_dirichlet_alpha = None if args.disable_root_noise else args.root_dirichlet_alpha

    saved_paths = generate_mcts_selfplay_games(
        output_dir=Path(args.output_dir),
        checkpoint_path=args.checkpoint,
        games=args.games,
        max_plies=args.max_plies,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        num_workers=args.num_workers,
    )

    print()
    print("MCTS self-play generation")
    print("-" * 40)
    print(f"Checkpoint:       {args.checkpoint or 'fresh random model'}")
    print(f"Games:            {len(saved_paths)}")
    print(f"Max plies:        {args.max_plies}")
    print(f"Simulations/move: {args.num_simulations}")
    print(f"C-PUCT:           {args.c_puct}")
    print(f"Temperature:      {args.temperature}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Workers:          {args.num_workers}")
    print()

    for path in saved_paths[:10]:
        print(f"saved: {path}")

    if len(saved_paths) > 10:
        print(f"... {len(saved_paths) - 10} more")


if __name__ == "__main__":
    main()