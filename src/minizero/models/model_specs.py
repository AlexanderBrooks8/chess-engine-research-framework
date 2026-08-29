from __future__ import annotations

MODEL_TYPE = "transformer"

MODEL_CONFIG = {
    "d_model": 192,
    "n_layers": 4,
    "n_heads": 6,
    "ff_dim": 768,
    "dropout": 0.1,
}

CNN_CONFIG = {
    "channels": 128,
    "n_blocks": 6,
    "dropout": 0.1,
}