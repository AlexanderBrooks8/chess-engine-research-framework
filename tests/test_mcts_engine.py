import chess
import pytest

from minizero.engine.mcts_engine import MCTSEngine
from minizero.models.transformer_policy_value import TransformerPolicyValue


def make_model() -> TransformerPolicyValue:
    return TransformerPolicyValue(
        d_model=64,
        n_layers=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
    )


def test_mcts_engine_returns_legal_move():
    board = chess.Board()
    engine = MCTSEngine(
        model=make_model(),
        num_simulations=2,
        temperature=0.0,
    )

    move = engine.choose_move(board)

    assert move in board.legal_moves


def test_mcts_engine_name():
    engine = MCTSEngine(
        model=make_model(),
        num_simulations=2,
    )

    assert engine.name == "mcts"


def test_mcts_engine_rejects_model_and_checkpoint():
    with pytest.raises(ValueError):
        MCTSEngine(
            model=make_model(),
            checkpoint_path="fake.pt",
            num_simulations=2,
        )


def test_mcts_engine_rejects_nonpositive_simulations():
    with pytest.raises(ValueError):
        MCTSEngine(
            model=make_model(),
            num_simulations=0,
        )