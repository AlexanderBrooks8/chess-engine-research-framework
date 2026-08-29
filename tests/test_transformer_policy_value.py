import chess
import torch

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import VOCAB_SIZE
from minizero.models.transformer_policy_value import TransformerPolicyValue


def test_transformer_policy_value_output_shapes():
    board = chess.Board()
    encoded = encode_position(board)

    model = TransformerPolicyValue(
        d_model=64,
        n_layers=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
    )

    output = model(
        board_tokens=encoded["board_tokens"].unsqueeze(0),
        side_to_move=encoded["side_to_move"].unsqueeze(0),
        castling_rights=encoded["castling_rights"].unsqueeze(0),
        en_passant=encoded["en_passant"].unsqueeze(0),
    )

    assert output.policy_logits.shape == (1, VOCAB_SIZE)
    assert output.value.shape == (1,)


def test_transformer_value_is_bounded():
    board = chess.Board()
    encoded = encode_position(board)

    model = TransformerPolicyValue(
        d_model=64,
        n_layers=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
    )

    output = model(
        board_tokens=encoded["board_tokens"].unsqueeze(0),
        side_to_move=encoded["side_to_move"].unsqueeze(0),
        castling_rights=encoded["castling_rights"].unsqueeze(0),
        en_passant=encoded["en_passant"].unsqueeze(0),
    )

    value = float(output.value.item())

    assert -1.0 <= value <= 1.0


def test_transformer_supports_batch_input():
    board = chess.Board()
    encoded = encode_position(board)

    model = TransformerPolicyValue(
        d_model=64,
        n_layers=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
    )

    board_tokens = torch.stack(
        [
            encoded["board_tokens"],
            encoded["board_tokens"],
        ]
    )

    side_to_move = torch.stack(
        [
            encoded["side_to_move"],
            encoded["side_to_move"],
        ]
    )

    castling_rights = torch.stack(
        [
            encoded["castling_rights"],
            encoded["castling_rights"],
        ]
    )

    en_passant = torch.stack(
        [
            encoded["en_passant"],
            encoded["en_passant"],
        ]
    )

    output = model(
        board_tokens=board_tokens,
        side_to_move=side_to_move,
        castling_rights=castling_rights,
        en_passant=en_passant,
    )

    assert output.policy_logits.shape == (2, VOCAB_SIZE)
    assert output.value.shape == (2,)