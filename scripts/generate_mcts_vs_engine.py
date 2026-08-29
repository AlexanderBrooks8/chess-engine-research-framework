from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from minizero.engine.material_engine import MaterialEngine
from minizero.engine.random_engine import RandomEngine
from minizero.engine.tactical_engine import TacticalEngine
from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.transformer_policy_value import TransformerPolicyValue
from minizero.search.mcts import run_mcts
from minizero.selfplay.game_record import GameRecord, save_game_record
from minizero.models.model_specs import MODEL_CONFIG


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def material_score(board: chess.Board) -> float:
    score = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            continue

        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value

    return score


def adjudicate_result_by_material(
    board: chess.Board,
    threshold: float = 1.0,
) -> str:
    score = material_score(board)

    if score >= threshold:
        return "1-0"

    if score <= -threshold:
        return "0-1"

    return "1/2-1/2"


def make_model(checkpoint: str | None) -> TransformerPolicyValue:
    if checkpoint is None:
        model = TransformerPolicyValue(**MODEL_CONFIG)
    else:
        model = load_transformer_from_checkpoint(checkpoint)

    model.eval()
    return model


def make_opponent(name: str):
    if name == "random":
        return RandomEngine()

    if name == "material":
        return MaterialEngine()

    if name == "tactical":
        return TacticalEngine()

    raise ValueError(f"Unsupported opponent: {name}")


def model_color_for_game(
    game_idx: int,
    model_color: str,
) -> bool:
    if model_color == "white":
        return chess.WHITE

    if model_color == "black":
        return chess.BLACK

    if model_color == "both":
        return chess.WHITE if game_idx % 2 == 0 else chess.BLACK

    raise ValueError(f"Unsupported model color: {model_color}")


def play_mcts_vs_engine_game(
    model: TransformerPolicyValue,
    opponent_name: str,
    game_idx: int,
    max_plies: int,
    num_simulations: int,
    c_puct: float,
    temperature: float,
    model_color: str,
    root_dirichlet_alpha: float | None,
    root_exploration_fraction: float,
    adjudication_threshold: float,
) -> GameRecord:
    opponent = make_opponent(opponent_name)
    mcts_color = model_color_for_game(
        game_idx=game_idx,
        model_color=model_color,
    )

    board = chess.Board()
    record = GameRecord()
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == mcts_color:
            result = run_mcts(
                board=board,
                model=model,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature=temperature,
                root_dirichlet_alpha=root_dirichlet_alpha,
                root_exploration_fraction=root_exploration_fraction,
            )

            move = result.move

            if move not in board.legal_moves:
                raise ValueError(f"MCTS produced illegal move: {move}")

            record.add_position(
                board,
                move,
                policy_target=result.policy_target,
            )
        else:
            move = opponent.choose_move(board)

            if move not in board.legal_moves:
                raise ValueError(f"{opponent.name} produced illegal move: {move}")

        board.push(move)
        plies += 1

    if board.is_game_over(claim_draw=True):
        result_string = board.result(claim_draw=True)
    else:
        result_string = adjudicate_result_by_material(
            board,
            threshold=adjudication_threshold,
        )

    record.finalize(result_string)
    return record


def generate_mcts_vs_engine_games(
    output_dir: str | Path,
    checkpoint: str | None,
    opponent: str,
    games: int,
    max_plies: int,
    num_simulations: int,
    c_puct: float,
    temperature: float,
    model_color: str,
    root_dirichlet_alpha: float | None,
    root_exploration_fraction: float,
    adjudication_threshold: float,
) -> list[Path]:
    if games <= 0:
        raise ValueError("games must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(checkpoint)
    saved_paths: list[Path] = []

    for game_idx in range(games):
        record = play_mcts_vs_engine_game(
            model=model,
            opponent_name=opponent,
            game_idx=game_idx,
            max_plies=max_plies,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            model_color=model_color,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            adjudication_threshold=adjudication_threshold,
        )

        path = output_dir / f"mcts_vs_{opponent}_{game_idx:06d}.pt"
        save_game_record(path, record)
        saved_paths.append(path)
        #print(f"saved: {path}")

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MCTS-vs-engine curriculum games.")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--opponent", choices=["random", "material", "tactical"], default="random")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model-color", choices=["white", "black", "both"], default="both")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--adjudication-threshold", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_num_threads(1)

    root_dirichlet_alpha = (
        None if args.root_dirichlet_alpha <= 0 else args.root_dirichlet_alpha
    )

    print()
    print("MCTS-vs-engine curriculum generation")
    print("-" * 40)
    print(f"Checkpoint:       {args.checkpoint or 'fresh random model'}")
    print(f"Opponent:         {args.opponent}")
    print(f"Games:            {args.games}")
    print(f"Model color:      {args.model_color}")
    print(f"Max plies:        {args.max_plies}")
    print(f"Simulations/move: {args.num_simulations}")
    print(f"C-PUCT:           {args.c_puct}")
    print(f"Temperature:      {args.temperature}")
    print(f"Root alpha:       {root_dirichlet_alpha}")
    print(f"Output dir:       {args.output_dir}")
    print()

    generate_mcts_vs_engine_games(
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        opponent=args.opponent,
        games=args.games,
        max_plies=args.max_plies,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        model_color=args.model_color,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        adjudication_threshold=args.adjudication_threshold,
    )


if __name__ == "__main__":
    main()