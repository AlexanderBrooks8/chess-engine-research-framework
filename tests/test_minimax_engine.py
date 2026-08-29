import chess
import pytest

from minizero.engine.minimax_engine import (
    MinimaxEngine,
    evaluate_board,
    material_score_white_perspective,
)


def test_material_score_white_perspective_starting_position_is_equal():
    board = chess.Board()

    assert material_score_white_perspective(board) == 0.0


def test_evaluate_board_returns_negative_for_checkmated_side_to_move():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert board.is_checkmate()
    assert evaluate_board(board) == float("-inf")


def test_minimax_engine_finds_mate_in_one():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1")
    engine = MinimaxEngine(depth=1)

    move = engine.choose_move(board)

    board.push(move)
    assert board.is_checkmate()


def test_minimax_engine_rejects_non_positive_depth():
    with pytest.raises(ValueError):
        MinimaxEngine(depth=0)


def test_minimax_engine_returns_legal_move_from_start_position():
    board = chess.Board()
    engine = MinimaxEngine(depth=1)

    move = engine.choose_move(board)

    assert move in board.legal_moves
