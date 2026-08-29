from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, TypeVar

from minizero.engine.base import BaseEngine
from minizero.engine.random_engine import RandomEngine
from minizero.engine.material_engine import MaterialEngine
from minizero.engine.tactical_engine import TacticalEngine
from minizero.engine.tactical_neural_engine import TacticalNeuralEngine
from minizero.engine.neural_engine import NeuralEngine
from minizero.engine.neural_search_engine import NeuralSearchEngine
from minizero.engine.mcts_engine import MCTSEngine
from minizero.engine.minimax_engine import MinimaxEngine
from minizero.eval.arena import load_opening_fens, run_match


T = TypeVar("T")


ENGINES: dict[str, Callable[[], BaseEngine]] = {
    "random": RandomEngine,
    "material": MaterialEngine,
    "tactical": TacticalEngine,
    "neural": NeuralEngine,
    "neural_sample": lambda: NeuralEngine(deterministic=False, temperature=1.0),
    "tactical_neural": lambda: TacticalNeuralEngine(),
    "neural_search": lambda: NeuralSearchEngine(),
    "mcts": lambda: MCTSEngine(),
    "minimax": lambda: MinimaxEngine(),
}


def override_or_default(override: T | None, default: T) -> T:
    return default if override is None else override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run engine-vs-engine arena matches.")
    parser.add_argument("--white", choices=ENGINES.keys(), default="material")
    parser.add_argument("--black", choices=ENGINES.keys(), default="random")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--pgn", type=str, default=None)
    parser.add_argument(
        "--opening-fens",
        type=str,
        default=None,
        help="Optional text file with one starting FEN per line. Games cycle through these FENs.",
    )
    parser.add_argument(
        "--checkpoint-white",
        type=str,
        default=None,
        help="Optional checkpoint path for the white neural engine.",
    )
    parser.add_argument(
        "--checkpoint-black",
        type=str,
        default=None,
        help="Optional checkpoint path for the black neural engine.",
    )

    parser.add_argument("--adjudicate-material", action="store_true")
    parser.add_argument("--adjudication-threshold", type=float, default=1.0)
    parser.add_argument("--mcts-sims", type=int, default=32)
    parser.add_argument("--mcts-time-ms", type=float, default=None)
    parser.add_argument("--mcts-temperature", type=float, default=0.0)
    parser.add_argument("--mcts-c-puct", type=float, default=1.5)
    parser.add_argument("--mcts-batch-size", type=int, default=1)
    parser.add_argument("--mcts-max-children", type=int, default=None)
    parser.add_argument("--mcts-tactical-veto", action="store_true")
    parser.add_argument("--mcts-veto-top-n", type=int, default=8)
    parser.add_argument("--mcts-veto-depth", type=int, default=2)
    parser.add_argument("--mcts-veto-threshold-pawns", type=float, default=2.0)
    parser.add_argument("--mcts-veto-mobility-weight", type=float, default=0.01)
    parser.add_argument("--mcts-device", type=str, default="cpu")
    parser.add_argument("--disable-mcts-cache", action="store_true")
    parser.add_argument("--neural-device", type=str, default="cpu")
    parser.add_argument("--minimax-depth", type=int, default=2)
    parser.add_argument("--minimax-mobility-weight", type=float, default=0.01)

    parser.add_argument("--search-root-top-k", type=int, default=8)
    parser.add_argument("--search-reply-top-k", type=int, default=4)
    parser.add_argument("--search-depth", type=int, choices=[1, 2], default=2)
    parser.add_argument("--search-policy-weight", type=float, default=0.25)
    parser.add_argument("--search-value-weight", type=float, default=1.0)
    parser.add_argument("--disable-search-cache", action="store_true")

    parser.add_argument("--search-root-top-k-white", type=int, default=None)
    parser.add_argument("--search-reply-top-k-white", type=int, default=None)
    parser.add_argument("--search-depth-white", type=int, choices=[1, 2], default=None)
    parser.add_argument("--search-policy-weight-white", type=float, default=None)
    parser.add_argument("--search-value-weight-white", type=float, default=None)

    parser.add_argument("--search-root-top-k-black", type=int, default=None)
    parser.add_argument("--search-reply-top-k-black", type=int, default=None)
    parser.add_argument("--search-depth-black", type=int, choices=[1, 2], default=None)
    parser.add_argument("--search-policy-weight-black", type=float, default=None)
    parser.add_argument("--search-value-weight-black", type=float, default=None)

    return parser.parse_args()


def build_engine(
    engine_name: str,
    checkpoint_path: str | None = None,
    mcts_sims: int = 32,
    mcts_time_ms: float | None = None,
    mcts_temperature: float = 0.0,
    mcts_c_puct: float = 1.5,
    mcts_batch_size: int = 1,
    mcts_max_children: int | None = None,
    mcts_tactical_veto: bool = False,
    mcts_veto_top_n: int = 8,
    mcts_veto_depth: int = 2,
    mcts_veto_threshold_pawns: float = 2.0,
    mcts_veto_mobility_weight: float = 0.01,
    mcts_device: str = "cpu",
    disable_mcts_cache: bool = False,
    neural_device: str = "cpu",
    minimax_depth: int = 2,
    minimax_mobility_weight: float = 0.01,
    search_root_top_k: int = 8,
    search_reply_top_k: int = 4,
    search_depth: int = 2,
    search_policy_weight: float = 0.25,
    search_value_weight: float = 1.0,
    disable_search_cache: bool = False,
):
    if engine_name == "minimax":
        return MinimaxEngine(
            depth=minimax_depth,
            mobility_weight=minimax_mobility_weight,
        )

    if engine_name == "mcts":
        return MCTSEngine(
            checkpoint_path=checkpoint_path,
            num_simulations=mcts_sims,
            time_limit_ms=mcts_time_ms,
            c_puct=mcts_c_puct,
            temperature=mcts_temperature,
            leaf_batch_size=mcts_batch_size,
            max_children=mcts_max_children,
            tactical_veto=mcts_tactical_veto,
            veto_top_n=mcts_veto_top_n,
            veto_depth=mcts_veto_depth,
            veto_threshold_pawns=mcts_veto_threshold_pawns,
            veto_mobility_weight=mcts_veto_mobility_weight,
            device=mcts_device,
            use_eval_cache=not disable_mcts_cache,
        )

    if engine_name == "neural":
        return NeuralEngine(checkpoint_path=checkpoint_path, device=neural_device)

    if engine_name == "neural_sample":
        return NeuralEngine(
            checkpoint_path=checkpoint_path,
            deterministic=False,
            temperature=1.0,
            device=neural_device,
        )

    if engine_name == "neural_search":
        return NeuralSearchEngine(
            checkpoint_path=checkpoint_path,
            device=neural_device,
            root_top_k=search_root_top_k,
            reply_top_k=search_reply_top_k,
            depth=search_depth,
            policy_weight=search_policy_weight,
            value_weight=search_value_weight,
            use_eval_cache=not disable_search_cache,
        )

    if engine_name == "tactical_neural":
        return TacticalNeuralEngine(
            checkpoint_path=checkpoint_path,
            device=neural_device,
        )

    if checkpoint_path is None:
        return ENGINES[engine_name]()

    raise ValueError(
        "Checkpoint paths can only be used with neural, neural_sample, "
        "neural_search, tactical_neural, or mcts engines."
    )


def main() -> None:
    args = parse_args()

    white_engine = build_engine(
        engine_name=args.white,
        checkpoint_path=args.checkpoint_white,
        mcts_sims=args.mcts_sims,
        mcts_time_ms=args.mcts_time_ms,
        mcts_temperature=args.mcts_temperature,
        mcts_c_puct=args.mcts_c_puct,
        mcts_batch_size=args.mcts_batch_size,
        mcts_max_children=args.mcts_max_children,
        mcts_tactical_veto=args.mcts_tactical_veto,
        mcts_veto_top_n=args.mcts_veto_top_n,
        mcts_veto_depth=args.mcts_veto_depth,
        mcts_veto_threshold_pawns=args.mcts_veto_threshold_pawns,
        mcts_veto_mobility_weight=args.mcts_veto_mobility_weight,
        mcts_device=args.mcts_device,
        disable_mcts_cache=args.disable_mcts_cache,
        neural_device=args.neural_device,
        minimax_depth=args.minimax_depth,
        minimax_mobility_weight=args.minimax_mobility_weight,
        search_root_top_k=override_or_default(
            args.search_root_top_k_white,
            args.search_root_top_k,
        ),
        search_reply_top_k=override_or_default(
            args.search_reply_top_k_white,
            args.search_reply_top_k,
        ),
        search_depth=override_or_default(
            args.search_depth_white,
            args.search_depth,
        ),
        search_policy_weight=override_or_default(
            args.search_policy_weight_white,
            args.search_policy_weight,
        ),
        search_value_weight=override_or_default(
            args.search_value_weight_white,
            args.search_value_weight,
        ),
        disable_search_cache=args.disable_search_cache,
    )

    black_engine = build_engine(
        engine_name=args.black,
        checkpoint_path=args.checkpoint_black,
        mcts_sims=args.mcts_sims,
        mcts_time_ms=args.mcts_time_ms,
        mcts_temperature=args.mcts_temperature,
        mcts_c_puct=args.mcts_c_puct,
        mcts_batch_size=args.mcts_batch_size,
        mcts_max_children=args.mcts_max_children,
        mcts_tactical_veto=args.mcts_tactical_veto,
        mcts_veto_top_n=args.mcts_veto_top_n,
        mcts_veto_depth=args.mcts_veto_depth,
        mcts_veto_threshold_pawns=args.mcts_veto_threshold_pawns,
        mcts_veto_mobility_weight=args.mcts_veto_mobility_weight,
        mcts_device=args.mcts_device,
        disable_mcts_cache=args.disable_mcts_cache,
        neural_device=args.neural_device,
        minimax_depth=args.minimax_depth,
        minimax_mobility_weight=args.minimax_mobility_weight,
        search_root_top_k=override_or_default(
            args.search_root_top_k_black,
            args.search_root_top_k,
        ),
        search_reply_top_k=override_or_default(
            args.search_reply_top_k_black,
            args.search_reply_top_k,
        ),
        search_depth=override_or_default(
            args.search_depth_black,
            args.search_depth,
        ),
        search_policy_weight=override_or_default(
            args.search_policy_weight_black,
            args.search_policy_weight,
        ),
        search_value_weight=override_or_default(
            args.search_value_weight_black,
            args.search_value_weight,
        ),
        disable_search_cache=args.disable_search_cache,
    )

    pgn_path = (
        Path(args.pgn)
        if args.pgn is not None
        else Path("runs") / "arena" / f"{args.white}_vs_{args.black}.pgn"
    )

    opening_fens = (
        load_opening_fens(args.opening_fens)
        if args.opening_fens is not None
        else None
    )

    result = run_match(
        white_engine,
        black_engine,
        games=args.games,
        max_plies=args.max_plies,
        pgn_path=pgn_path,
        adjudicate_material=args.adjudicate_material,
        adjudication_threshold=args.adjudication_threshold,
        opening_fens=opening_fens,
    )

    print()
    print(f"{result.white_engine} vs {result.black_engine}")
    print("-" * 40)
    print(f"Games:      {result.games}")
    print(f"White wins: {result.white_wins}")
    print(f"Black wins: {result.black_wins}")
    print(f"Draws:      {result.draws}")
    print(f"Avg plies:  {result.avg_plies:.2f}")
    print()
    print("Terminations:")

    for termination, count in result.termination_counts.items():
        print(f"  {termination}: {count}")

    print()
    print(f"PGN saved to: {pgn_path}")
    print()


if __name__ == "__main__":
    main()