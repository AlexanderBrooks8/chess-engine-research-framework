from __future__ import annotations

import random
import chess


def choose_random_move(board: chess.Board) -> chess.Move:
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("No legal moves available.")

    return random.choice(legal_moves)