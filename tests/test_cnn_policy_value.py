import torch

from minizero.chess.move_vocab import VOCAB_SIZE
from minizero.models.cnn_policy_value import CNNPolicyValue
from minizero.models.factory import build_model_from_config


def test_cnn_policy_value_forward_shapes():
    model = CNNPolicyValue(channels=32, n_blocks=2, dropout=0.0)

    output = model(
        board_tokens=torch.zeros((2, 64), dtype=torch.long),
        side_to_move=torch.zeros((2,), dtype=torch.long),
        castling_rights=torch.zeros((2,), dtype=torch.long),
        en_passant=torch.full((2,), 64, dtype=torch.long),
    )

    assert output.policy_logits.shape == (2, VOCAB_SIZE)
    assert output.value.shape == (2,)
    assert output.material.shape == (2,)
    assert output.mate_in_1_logits.shape == (2,)
    assert output.in_check_logits.shape == (2,)
    assert output.has_check_logits.shape == (2,)
    assert output.capture_available_logits.shape == (2,)
    assert output.legal_mobility.shape == (2,)
    assert output.attack_pressure.shape == (2,)
    assert output.king_pressure.shape == (2,)
    assert output.best_capture.shape == (2,)
    assert output.hanging_material.shape == (2,)


def test_build_model_from_config_supports_cnn():
    model = build_model_from_config(
        {
            "model_type": "cnn",
            "channels": 32,
            "n_blocks": 2,
            "dropout": 0.0,
        }
    )

    assert isinstance(model, CNNPolicyValue)
