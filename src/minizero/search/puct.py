from __future__ import annotations

import math

from minizero.search.node import MCTSNode


def puct_score(
    parent: MCTSNode,
    child: MCTSNode,
    c_puct: float,
) -> float:
    prior_score = c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)

    # child.mean_value is from the child player's perspective.
    # From the parent player's perspective, that value has opposite sign.
    value_score = -child.mean_value

    return value_score + prior_score