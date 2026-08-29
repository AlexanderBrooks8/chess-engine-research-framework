from __future__ import annotations

from abc import ABC, abstractmethod
import chess


class BaseEngine(ABC):
    name: str

    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move:
        pass