from __future__ import annotations

import math

import chess

from minizero.engine.base import BaseEngine


PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def material_score_white_perspective(board: chess.Board) -> float:
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


def evaluate_board(
    board: chess.Board,
    mobility_weight: float = 0.01,
) -> float:
    """Return static evaluation from the side-to-move perspective."""
    if board.is_checkmate():
        return -math.inf

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    material = material_score_white_perspective(board)

    if board.turn == chess.BLACK:
        material = -material

    mobility = mobility_weight * board.legal_moves.count()
    return material + mobility


class MinimaxEngine(BaseEngine):
    """Classical minimax engine using negamax + alpha-beta pruning.

    Static evaluation uses material plus a small mobility bonus as a deterministic baseline.
    """

    name = "minimax"

    def __init__(
        self,
        depth: int = 2,
        mobility_weight: float = 0.01,
        checkmate_score: float = 10_000.0,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.depth = depth
        self.mobility_weight = mobility_weight
        self.checkmate_score = checkmate_score

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=False):
            raise ValueError("No legal moves available because the game is over.")

        best_score = -math.inf
        best_move: chess.Move | None = None

        # Sort for deterministic tie-breaking and mildly better alpha-beta behavior.
        legal_moves = sorted(board.legal_moves, key=self._move_sort_key)

        alpha = -math.inf
        beta = math.inf

        for move in legal_moves:
            board.push(move)
            score = -self._negamax(
                board=board,
                depth=self.depth - 1,
                alpha=-beta,
                beta=-alpha,
            )
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        if best_move is None:
            raise RuntimeError("MinimaxEngine failed to choose a move.")

        return best_move

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if board.is_checkmate():
            return -self.checkmate_score

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        if depth == 0:
            return evaluate_board(
                board,
                mobility_weight=self.mobility_weight,
            )

        best_score = -math.inf

        for move in sorted(board.legal_moves, key=self._move_sort_key):
            board.push(move)
            score = -self._negamax(
                board=board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
            )
            board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score

    @staticmethod
    def _move_sort_key(move: chess.Move) -> str:
        return move.uci()
