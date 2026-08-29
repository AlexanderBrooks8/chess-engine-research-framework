from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import chess

from minizero.engine.base import BaseEngine
from minizero.engine.neural_engine import NeuralEngine
from minizero.engine.tactical_neural_engine import TacticalNeuralEngine
from minizero.engine.neural_search_engine import NeuralSearchEngine

EngineKind = Literal["neural", "tactical_neural"]


@dataclass(frozen=True)
class MiniZeroBotConfig:
    checkpoint_path: Path
    engine_kind: str
    device: str
    search_depth: int = 2
    search_root_top_k: int = 8
    search_reply_top_k: int = 4
    search_policy_weight: float = 0.25
    search_value_weight: float = 1.0
    search_policy_temperature: float = 1.0
    use_eval_cache: bool = True


def build_minizero_engine(config: MiniZeroBotConfig) -> BaseEngine:
    if config.engine_kind == "neural":
        return NeuralEngine(
            checkpoint_path=config.checkpoint_path,
            device=config.device,
        )

    if config.engine_kind == "tactical_neural":
        return TacticalNeuralEngine(
            checkpoint_path=config.checkpoint_path,
            device=config.device,
        )

    if config.engine_kind == "neural_search":
        return NeuralSearchEngine(
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            search_depth=config.search_depth,
            root_top_k=config.search_root_top_k,
            reply_top_k=config.search_reply_top_k,
            policy_weight=config.search_policy_weight,
            value_weight=config.search_value_weight,
            policy_temperature=config.search_policy_temperature,
            use_eval_cache=config.use_eval_cache,
        )

    raise ValueError(f"Unsupported MiniZero engine kind: {config.engine_kind!r}")


def normalize_root_moves(root_moves: object) -> list[chess.Move] | None:
    if isinstance(root_moves, list):
        return [move for move in root_moves if isinstance(move, chess.Move)]
    return None


def choose_legal_or_root_move(
    engine: BaseEngine,
    board: chess.Board,
    root_moves: object = None,
) -> chess.Move:
    """Choose a legal move, respecting lichess-bot root move restrictions.

    lichess-bot can pass root_moves when tablebases/opening books restrict the
    allowed move set. The MiniZero engines currently score from the full legal
    move set, so this helper uses the engine's preferred move when allowed and
    falls back deterministically to the first root move when it is not allowed.
    """
    allowed_root_moves = normalize_root_moves(root_moves)
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available.")

    if allowed_root_moves == []:
        raise ValueError("root_moves was an empty list; no move can be selected.")

    move = engine.choose_move(board)
    if move not in legal_moves:
        raise ValueError(f"MiniZero engine produced illegal move: {move}")

    if allowed_root_moves is None or move in allowed_root_moves:
        return move

    legal_allowed = [move for move in allowed_root_moves if move in legal_moves]
    if not legal_allowed:
        raise ValueError("root_moves contained no legal moves for the current board.")

    return sorted(legal_allowed, key=str)[0]


def parse_engine_kind(value: str) -> str:
    value = value.strip().lower()

    valid = {
        "neural",
        "tactical_neural",
        "neural_search",
    }

    if value not in valid:
        raise ValueError(f"Unknown MINIZERO_ENGINE={value!r}. Valid options: {sorted(valid)}")

    return value
