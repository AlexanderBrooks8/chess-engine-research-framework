from __future__ import annotations

import chess
import pytest
import torch

from minizero.engine.neural_search_engine import NeuralSearchEngine
from minizero.search.mcts import MCTSEvaluation


class DummyModel(torch.nn.Module):
    def forward(self, **kwargs):  # pragma: no cover - evaluate_positions is monkeypatched.
        raise AssertionError("model forward should not be called in this unit test")


@pytest.fixture
def engine() -> NeuralSearchEngine:
    return NeuralSearchEngine(
        model=DummyModel(),
        device="cpu",
        root_top_k=2,
        reply_top_k=1,
        depth=1,
        policy_weight=0.0,
        value_weight=1.0,
        use_eval_cache=False,
    )


def test_neural_search_depth1_prefers_higher_child_value(monkeypatch: pytest.MonkeyPatch, engine: NeuralSearchEngine) -> None:
    board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")

    def fake_evaluate_positions(boards, model, cache=None):
        if len(boards) == 1 and boards[0].move_stack == []:
            return [MCTSEvaluation(value=0.0, move_priors={e4: 0.6, d4: 0.4})]

        out = []
        for child in boards:
            # Child values are from opponent side-to-move perspective.
            if child.peek() == e4:
                out.append(MCTSEvaluation(value=-0.8, move_priors={}))
            elif child.peek() == d4:
                out.append(MCTSEvaluation(value=-0.2, move_priors={}))
            else:  # pragma: no cover
                raise AssertionError(f"unexpected child board: {child.peek()}")
        return out

    monkeypatch.setattr("minizero.engine.neural_search_engine.evaluate_positions", fake_evaluate_positions)

    assert engine.choose_move(board) == e4


def test_neural_search_depth2_uses_worst_opponent_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    e5 = chess.Move.from_uci("e7e5")
    d5 = chess.Move.from_uci("d7d5")

    engine = NeuralSearchEngine(
        model=DummyModel(),
        device="cpu",
        root_top_k=2,
        reply_top_k=1,
        depth=2,
        policy_weight=0.0,
        value_weight=1.0,
        use_eval_cache=False,
    )

    def fake_evaluate_positions(boards, model, cache=None):
        if len(boards) == 1 and boards[0].move_stack == []:
            return [MCTSEvaluation(value=0.0, move_priors={e4: 0.6, d4: 0.4})]

        # Child boards after root moves: provide opponent reply priors.
        if all(len(child.move_stack) == 1 for child in boards):
            out = []
            for child in boards:
                if child.peek() == e4:
                    out.append(MCTSEvaluation(value=-0.9, move_priors={e5: 1.0}))
                elif child.peek() == d4:
                    out.append(MCTSEvaluation(value=-0.1, move_priors={d5: 1.0}))
                else:  # pragma: no cover
                    raise AssertionError(f"unexpected child board: {child.peek()}")
            return out

        # Reply boards after two plies: values are from root side-to-move perspective.
        out = []
        for reply_board in boards:
            moves = reply_board.move_stack
            if moves[-2:] == [e4, e5]:
                out.append(MCTSEvaluation(value=-0.7, move_priors={}))
            elif moves[-2:] == [d4, d5]:
                out.append(MCTSEvaluation(value=0.3, move_priors={}))
            else:  # pragma: no cover
                raise AssertionError(f"unexpected reply line: {moves}")
        return out

    monkeypatch.setattr("minizero.engine.neural_search_engine.evaluate_positions", fake_evaluate_positions)

    assert engine.choose_move(board) == d4


def test_neural_search_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        NeuralSearchEngine(model=DummyModel(), depth=3)
