# src/minizero/models/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PolicyValueOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor


class PolicyValueModel(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        board_tokens: torch.Tensor,
        side_to_move: torch.Tensor,
        castling_rights: torch.Tensor,
        en_passant: torch.Tensor,
    ) -> PolicyValueOutput:
        pass