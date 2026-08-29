from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess

from minizero.engine.base import BaseEngine
from minizero.engine.material_engine import MaterialEngine
from minizero.engine.neural_engine import NeuralEngine
from minizero.engine.random_engine import RandomEngine
from minizero.engine.tactical_engine import TacticalEngine
from minizero.selfplay.game_record import GameRecord, save_game_record


EngineFactory = Callable[[], BaseEngine]


ENGINES: dict[str, EngineFactory] = {
    "random": RandomEngine,
    "material": MaterialEngine,
    "tactical": TacticalEngine,
    "neural": NeuralEngine,
    "neural_sample": lambda: NeuralEngine(deterministic=False, temperature=1.0),
}


@dataclass(frozen=True)
class SelfPlayGameResult:
    record: GameRecord
    result: str
    plies: int
    termination: str


def play_selfplay_game(
    white_engine: BaseEngine,
    black_engine: BaseEngine,
    max_plies: int = 256,
) -> SelfPlayGameResult:
    board = chess.Board()
    record = GameRecord()
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = engine.choose_move(board)

        if move not in board.legal_moves:
            raise ValueError(f"{engine.name} produced illegal move: {move}")

        record.add_position(board, move)
        board.push(move)
        plies += 1

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = board.result(claim_draw=True)
        termination = str(outcome.termination) if outcome is not None else "unknown"
    else:
        result = "1/2-1/2"
        termination = "max_plies"

    record.finalize(result)

    return SelfPlayGameResult(
        record=record,
        result=result,
        plies=plies,
        termination=termination,
    )


def build_engine(
    engine_name: str,
    checkpoint_path: str | Path | None = None,
) -> BaseEngine:
    if checkpoint_path is None:
        return ENGINES[engine_name]()

    if engine_name == "neural":
        return NeuralEngine(checkpoint_path=checkpoint_path)

    if engine_name == "neural_sample":
        return NeuralEngine(
            checkpoint_path=checkpoint_path,
            deterministic=False,
            temperature=1.0,
        )

    raise ValueError("Checkpoint paths can only be used with neural engines.")


def generate_selfplay_games(
    output_dir: str | Path,
    white: str = "neural_sample",
    black: str = "neural_sample",
    games: int = 10,
    max_plies: int = 256,
    checkpoint_white: str | Path | None = None,
    checkpoint_black: str | Path | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for game_idx in range(games):
        white_engine = build_engine(white, checkpoint_white)
        black_engine = build_engine(black, checkpoint_black)

        result = play_selfplay_game(
            white_engine=white_engine,
            black_engine=black_engine,
            max_plies=max_plies,
        )

        path = output_dir / f"game_{game_idx:06d}.pt"
        save_game_record(path, result.record)
        saved_paths.append(path)

    return saved_paths