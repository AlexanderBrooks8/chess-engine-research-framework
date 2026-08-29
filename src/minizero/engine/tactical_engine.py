from __future__ import annotations

import random

import chess

from minizero.agents.material_agent import evaluate_material
from minizero.engine.base import BaseEngine


class TacticalEngine(BaseEngine):
    name = "tactical"

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            raise ValueError("No legal moves available.")

        checkmate_moves: list[chess.Move] = []

        for move in legal_moves:
            board.push(move)
            is_checkmate = board.is_checkmate()
            board.pop()

            if is_checkmate:
                checkmate_moves.append(move)

        if checkmate_moves:
            return random.choice(checkmate_moves)

        maximizing_white = board.turn == chess.WHITE
        best_score = float("-inf") if maximizing_white else float("inf")
        best_moves: list[chess.Move] = []

        for move in legal_moves:
            board.push(move)
            score = evaluate_material(board)
            board.pop()

            if maximizing_white:
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)
            else:
                if score < best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        return random.choice(best_moves)