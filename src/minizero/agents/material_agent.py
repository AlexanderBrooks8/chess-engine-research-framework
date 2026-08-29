from __future__ import annotations

import chess


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def evaluate_material(board: chess.Board) -> float:
    score = 0.0

    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    return score


def choose_material_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available.")

    maximizing_white = board.turn == chess.WHITE

    best_move = legal_moves[0]
    best_score = float("-inf") if maximizing_white else float("inf")

    for move in legal_moves:
        board.push(move)
        score = evaluate_material(board)
        board.pop()

        if maximizing_white and score > best_score:
            best_score = score
            best_move = move
        elif not maximizing_white and score < best_score:
            best_score = score
            best_move = move

    return best_move