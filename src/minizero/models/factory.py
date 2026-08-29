from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minizero.models.cnn_policy_value import CNNPolicyValue
from minizero.models.transformer_policy_value import TransformerPolicyValue


ModelT = TransformerPolicyValue | CNNPolicyValue


def build_model_from_config(model_config: dict[str, Any]) -> ModelT:
    model_type = str(model_config.get("model_type", "transformer")).lower()

    if model_type == "transformer":
        transformer_config = {
            key: value
            for key, value in model_config.items()
            if key in {"piece_vocab_size", "move_vocab_size", "d_model", "n_layers", "n_heads", "ff_dim", "dropout"}
        }
        return TransformerPolicyValue(**transformer_config)

    if model_type == "cnn":
        cnn_config = {
            key: value
            for key, value in model_config.items()
            if key in {"piece_vocab_size", "move_vocab_size", "channels", "n_blocks", "dropout"}
        }
        return CNNPolicyValue(**cnn_config)

    raise ValueError(f"Unknown model_type: {model_type!r}")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> ModelT:
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=map_location,
        weights_only=False,
    )

    model_config: dict[str, Any] = checkpoint.get("model_config", {})
    model = build_model_from_config(model_config)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model

def load_transformer_from_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> ModelT:
    return load_model_from_checkpoint(checkpoint_path=checkpoint_path, map_location=map_location)