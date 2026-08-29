from __future__ import annotations

import torch
from torch import nn

from minizero.chess.move_vocab import VOCAB_SIZE
from minizero.models.base import PolicyValueModel
from minizero.models.model_specs import CNN_CONFIG
from minizero.models.transformer_policy_value import TransformerPolicyValueOutput


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class CNNPolicyValue(PolicyValueModel):
    """Small AlphaZero-style residual CNN over an 8x8 board.

    This model keeps the same call signature/output attributes as
    TransformerPolicyValue, so existing neural/MCTS/training code can switch
    architectures through the factory and model_config only.
    """

    def __init__(
        self,
        piece_vocab_size: int = 13,
        move_vocab_size: int = VOCAB_SIZE,
        channels: int = CNN_CONFIG["channels"],
        n_blocks: int = CNN_CONFIG["n_blocks"],
        dropout: float = CNN_CONFIG["dropout"],
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive.")

        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive.")

        self.piece_embedding = nn.Embedding(piece_vocab_size, channels)
        self.side_embedding = nn.Embedding(2, channels)
        self.castling_embedding = nn.Embedding(16, channels)
        self.en_passant_embedding = nn.Embedding(65, channels)

        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, move_vocab_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_dropout = nn.Dropout(dropout)

        self.material_head = self._scalar_regression_head(channels, tanh=True)
        self.mate_in_1_head = self._scalar_logit_head(channels)
        self.in_check_head = self._scalar_logit_head(channels)
        self.has_check_head = self._scalar_logit_head(channels)
        self.capture_available_head = self._scalar_logit_head(channels)
        self.legal_mobility_head = self._scalar_regression_head(channels, sigmoid=True)
        self.attack_pressure_head = self._scalar_regression_head(channels, sigmoid=True)
        self.king_pressure_head = self._scalar_regression_head(channels, sigmoid=True)
        self.best_capture_head = self._scalar_regression_head(channels, sigmoid=True)
        self.hanging_material_head = self._scalar_regression_head(channels, sigmoid=True)

    @staticmethod
    def _scalar_logit_head(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 1),
        )

    @staticmethod
    def _scalar_regression_head(
        channels: int,
        *,
        tanh: bool = False,
        sigmoid: bool = False,
    ) -> nn.Module:
        layers: list[nn.Module] = [
            nn.LayerNorm(channels),
            nn.Linear(channels, 1),
        ]
        if tanh:
            layers.append(nn.Tanh())
        if sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

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

        x = self.piece_embedding(board_tokens)  # [B, 64, C]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, 8, 8)

        meta = (
            self.side_embedding(side_to_move)
            + self.castling_embedding(castling_rights)
            + self.en_passant_embedding(en_passant)
        )
        x = x + meta[:, :, None, None]

        x = self.stem(x)
        x = self.residual_tower(x)

        pooled = self.global_pool(x).flatten(1)
        pooled = self.aux_dropout(pooled)

        return TransformerPolicyValueOutput(
            policy_logits=self.policy_head(x),
            value=self.value_head(x).squeeze(-1),
            material=self.material_head(pooled).squeeze(-1),
            mate_in_1_logits=self.mate_in_1_head(pooled).squeeze(-1),
            in_check_logits=self.in_check_head(pooled).squeeze(-1),
            has_check_logits=self.has_check_head(pooled).squeeze(-1),
            capture_available_logits=self.capture_available_head(pooled).squeeze(-1),
            legal_mobility=self.legal_mobility_head(pooled).squeeze(-1),
            attack_pressure=self.attack_pressure_head(pooled).squeeze(-1),
            king_pressure=self.king_pressure_head(pooled).squeeze(-1),
            best_capture=self.best_capture_head(pooled).squeeze(-1),
            hanging_material=self.hanging_material_head(pooled).squeeze(-1),
        )
