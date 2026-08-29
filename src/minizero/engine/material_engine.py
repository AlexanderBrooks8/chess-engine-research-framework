from __future__ import annotations

import chess

from minizero.agents.material_agent import choose_material_move
from minizero.engine.base import BaseEngine


class MaterialEngine(BaseEngine):
    name = "material"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return choose_material_move(board)