from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minizero.chess.move_vocab import VOCAB_SIZE
from minizero.models.base import PolicyValueModel
from minizero.models.model_specs import MODEL_CONFIG


@dataclass(frozen=True)
class TransformerPolicyValueOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    material: torch.Tensor | None = None
    mate_in_1_logits: torch.Tensor | None = None
    in_check_logits: torch.Tensor | None = None
    has_check_logits: torch.Tensor | None = None
    capture_available_logits: torch.Tensor | None = None
    legal_mobility: torch.Tensor | None = None
    attack_pressure: torch.Tensor | None = None
    king_pressure: torch.Tensor | None = None
    best_capture: torch.Tensor | None = None
    hanging_material: torch.Tensor | None = None


class TransformerPolicyValue(PolicyValueModel):
    def __init__(
        self,
        piece_vocab_size: int = 13,
        move_vocab_size: int = VOCAB_SIZE,
        d_model: int = MODEL_CONFIG["d_model"],
        n_layers: int = MODEL_CONFIG["n_layers"],
        n_heads: int = MODEL_CONFIG["n_heads"],
        ff_dim: int = MODEL_CONFIG["ff_dim"],
        dropout: float = MODEL_CONFIG["dropout"],
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.piece_embedding = nn.Embedding(piece_vocab_size, d_model)
        self.square_embedding = nn.Embedding(64, d_model)

        self.side_embedding = nn.Embedding(2, d_model)
        self.castling_embedding = nn.Embedding(16, d_model)
        self.en_passant_embedding = nn.Embedding(65, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, move_vocab_size),
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

        self.material_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

        self.mate_in_1_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.in_check_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.has_check_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.capture_available_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.legal_mobility_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.attack_pressure_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.king_pressure_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.best_capture_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.hanging_material_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        board_tokens: torch.Tensor,
        side_to_move: torch.Tensor,
        castling_rights: torch.Tensor,
        en_passant: torch.Tensor,
    ) -> TransformerPolicyValueOutput:
        if board_tokens.ndim != 2:
            raise ValueError("board_tokens must have shape [batch_size, 64].")

        if board_tokens.shape[1] != 64:
            raise ValueError("board_tokens must contain exactly 64 square tokens.")

        batch_size = board_tokens.shape[0]
        device = board_tokens.device

        square_ids = torch.arange(64, device=device).unsqueeze(0).expand(batch_size, 64)

        piece_x = self.piece_embedding(board_tokens)
        square_x = self.square_embedding(square_ids)
        board_x = piece_x + square_x

        meta_x = (
            self.side_embedding(side_to_move)
            + self.castling_embedding(castling_rights)
            + self.en_passant_embedding(en_passant)
        )

        cls_x = self.cls_token.expand(batch_size, 1, -1) + meta_x.unsqueeze(1)

        x = torch.cat([cls_x, board_x], dim=1)
        x = self.encoder(x)

        cls_output = x[:, 0]

        policy_logits = self.policy_head(cls_output)
        value = self.value_head(cls_output).squeeze(-1)
        material = self.material_head(cls_output).squeeze(-1)
        mate_in_1_logits = self.mate_in_1_head(cls_output).squeeze(-1)
        in_check_logits = self.in_check_head(cls_output).squeeze(-1)
        has_check_logits = self.has_check_head(cls_output).squeeze(-1)
        capture_available_logits = self.capture_available_head(cls_output).squeeze(-1)
        legal_mobility = self.legal_mobility_head(cls_output).squeeze(-1)
        attack_pressure = self.attack_pressure_head(cls_output).squeeze(-1)
        king_pressure = self.king_pressure_head(cls_output).squeeze(-1)
        best_capture = self.best_capture_head(cls_output).squeeze(-1)
        hanging_material = self.hanging_material_head(cls_output).squeeze(-1)

        return TransformerPolicyValueOutput(
            policy_logits=policy_logits,
            value=value,
            material=material,
            mate_in_1_logits=mate_in_1_logits,
            in_check_logits=in_check_logits,
            has_check_logits=has_check_logits,
            capture_available_logits=capture_available_logits,
            legal_mobility=legal_mobility,
            attack_pressure=attack_pressure,
            king_pressure=king_pressure,
            best_capture=best_capture,
            hanging_material=hanging_material,
        )