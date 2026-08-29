from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import chess
import torch

from minizero.eval.arena import load_opening_fens
from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.model_specs import MODEL_CONFIG
from minizero.models.transformer_policy_value import TransformerPolicyValue
from minizero.search.mcts import MCTSEvaluationCache, run_mcts


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_boards(opening_fens: Path | None, positions: int) -> list[chess.Board]:
    if opening_fens is not None:
        fens = load_opening_fens(opening_fens)
        return [chess.Board(fens[i % len(fens)]) for i in range(positions)]

    return [chess.Board() for _ in range(positions)]


def make_model(checkpoint: Path | None, device: torch.device) -> TransformerPolicyValue:
    if checkpoint is None:
        model = TransformerPolicyValue(**MODEL_CONFIG)
    else:
        model = load_transformer_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location=device,
        )

    model.to(device)
    model.eval()
    return model


def benchmark_config(
    boards: list[chess.Board],
    model: TransformerPolicyValue,
    num_simulations: int | None,
    time_limit_ms: float | None,
    c_puct: float,
    temperature: float,
    use_cache: bool,
    cache_size: int,
    leaf_batch_size: int,
    max_children: int | None,
) -> None:
    cache = MCTSEvaluationCache(max_size=cache_size) if use_cache else None
    elapsed_samples: list[float] = []
    simulations = 0
    model_evals = 0
    model_batches = 0
    weighted_batch_size_sum = 0.0
    max_model_batch_size = 0
    cache_hits = 0
    cache_misses = 0

    for board in boards:
        start = time.perf_counter()
        result = run_mcts(
            board=board,
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            time_limit_s=None if time_limit_ms is None else time_limit_ms / 1000.0,
            eval_cache=cache,
            leaf_batch_size=leaf_batch_size,
            max_children=max_children,
        )
        elapsed = time.perf_counter() - start

        elapsed_samples.append(elapsed)
        simulations += result.simulations_run
        model_evals += result.model_evaluations
        model_batches += result.model_batches
        weighted_batch_size_sum += result.avg_model_batch_size * result.model_batches
        max_model_batch_size = max(max_model_batch_size, result.max_model_batch_size)
        cache_hits += result.cache_hits
        cache_misses += result.cache_misses

    total_elapsed = sum(elapsed_samples)
    games = len(boards)

    label = (
        f"sims={num_simulations}"
        if time_limit_ms is None
        else f"time_ms={time_limit_ms:g}"
    )

    print()
    print(f"MCTS benchmark ({label})")
    print("-" * 40)
    print(f"Positions:            {games}")
    print(f"Total seconds:        {total_elapsed:.4f}")
    print(f"Avg ms/move:          {(total_elapsed / games) * 1000.0:.2f}")
    print(f"Median ms/move:       {statistics.median(elapsed_samples) * 1000.0:.2f}")
    print(f"Simulations:          {simulations}")
    print(f"Avg sims/move:        {simulations / games:.2f}")
    print(f"Sims/sec:             {simulations / total_elapsed:.2f}")
    avg_model_batch_size = (
        weighted_batch_size_sum / model_batches
        if model_batches > 0
        else 0.0
    )

    print(f"Model evaluations:    {model_evals}")
    print(f"Avg evals/move:       {model_evals / games:.2f}")
    print(f"Model batches:        {model_batches}")
    print(f"Avg model batch size: {avg_model_batch_size:.2f}")
    print(f"Max model batch size: {max_model_batch_size}")
    print(f"Cache hits:           {cache_hits}")
    print(f"Cache misses:         {cache_misses}")

    if cache_hits + cache_misses > 0:
        hit_rate = cache_hits / (cache_hits + cache_misses)
        print(f"Cache hit rate:       {hit_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniZero MCTS search speed.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opening-fens", type=Path, default=None)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--mcts-sims", default="32,64,128")
    parser.add_argument("--mcts-time-ms", default="")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mcts-batch-size", default="1,8,16,32")
    parser.add_argument("--mcts-max-children", type=int, default=None)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = make_model(args.checkpoint, device=device)
    boards = load_boards(args.opening_fens, positions=args.positions)

    print()
    print("MiniZero MCTS benchmark")
    print("-" * 40)
    print(f"Checkpoint:      {args.checkpoint if args.checkpoint is not None else 'fresh random model'}")
    print(f"Device:          {device}")
    print(f"Positions:       {len(boards)}")
    print(f"Opening FENs:    {args.opening_fens}")
    print(f"Cache enabled:   {not args.disable_cache}")
    print(f"Max children:    {args.mcts_max_children}")

    batch_sizes = parse_int_list(args.mcts_batch_size)

    for leaf_batch_size in batch_sizes:
        print()
        print(f"Leaf batch size: {leaf_batch_size}")
        print("=" * 40)

        for sims in parse_int_list(args.mcts_sims):
            benchmark_config(
                boards=boards,
                model=model,
                num_simulations=sims,
                time_limit_ms=None,
                c_puct=args.c_puct,
                temperature=args.temperature,
                use_cache=not args.disable_cache,
                cache_size=args.cache_size,
                leaf_batch_size=leaf_batch_size,
                max_children=args.mcts_max_children,
            )

        if args.mcts_time_ms.strip():
            for time_ms in parse_float_list(args.mcts_time_ms):
                benchmark_config(
                    boards=boards,
                    model=model,
                    num_simulations=None,
                    time_limit_ms=time_ms,
                    c_puct=args.c_puct,
                    temperature=args.temperature,
                    use_cache=not args.disable_cache,
                    cache_size=args.cache_size,
                    leaf_batch_size=leaf_batch_size,
                    max_children=args.mcts_max_children,
                )


if __name__ == "__main__":
    main()
