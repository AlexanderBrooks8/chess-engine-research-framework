from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class MCTSNode:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def expand(self, move_priors: dict[chess.Move, float]) -> None:
        for move, prior in move_priors.items():
            if move not in self.children:
                self.children[move] = MCTSNode(prior=prior)