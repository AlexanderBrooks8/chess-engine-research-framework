from __future__ import annotations

import chess

from minizero.agents.random_agent import choose_random_move
from minizero.engine.base import BaseEngine


class RandomEngine(BaseEngine):
    name = "random"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return choose_random_move(board)