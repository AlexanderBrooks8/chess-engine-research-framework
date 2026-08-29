from __future__ import annotations

import chess
import pytest

from minizero.integrations.lichess_bot_adapter import (
    choose_legal_or_root_move,
    normalize_root_moves,
    parse_engine_kind,
)


class DummyEngine:
    def __init__(self, move: chess.Move) -> None:
        self.move = move

    def choose_move(self, board: chess.Board) -> chess.Move:  # noqa: ARG002
        return self.move


def test_parse_engine_kind_accepts_expected_values() -> None:
    assert parse_engine_kind("neural") == "neural"
    assert parse_engine_kind(" tactical_neural ") == "tactical_neural"


def test_parse_engine_kind_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        parse_engine_kind("mcts")


def test_normalize_root_moves_accepts_list_only() -> None:
    moves = [chess.Move.from_uci("e2e4")]
    assert normalize_root_moves(moves) == moves
    assert normalize_root_moves(None) is None
    assert normalize_root_moves("e2e4") is None


def test_choose_legal_or_root_move_uses_engine_move_when_allowed() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    engine = DummyEngine(move)

    chosen = choose_legal_or_root_move(engine, board, [move])

    assert chosen == move


def test_choose_legal_or_root_move_falls_back_to_root_move_when_needed() -> None:
    board = chess.Board()
    engine = DummyEngine(chess.Move.from_uci("e2e4"))
    allowed = [chess.Move.from_uci("g1f3"), chess.Move.from_uci("b1c3")]

    chosen = choose_legal_or_root_move(engine, board, allowed)

    assert chosen == chess.Move.from_uci("b1c3")


def test_choose_legal_or_root_move_rejects_empty_root_moves() -> None:
    board = chess.Board()
    engine = DummyEngine(chess.Move.from_uci("e2e4"))

    with pytest.raises(ValueError):
        choose_legal_or_root_move(engine, board, [])
