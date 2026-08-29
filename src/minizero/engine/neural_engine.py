# src/minizero/engine/neural_engine.py

from __future__ import annotations

from pathlib import Path

import chess
import torch

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import id_to_move, legal_move_mask
from minizero.engine.base import BaseEngine
from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.transformer_policy_value import TransformerPolicyValue


class NeuralEngine(BaseEngine):
    name = "neural"

    def __init__(
        self,
        model: TransformerPolicyValue | None = None,
        device: str | torch.device | None = None,
        deterministic: bool = True,
        temperature: float = 1.0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.device = torch.device(device or "cpu")

        if checkpoint_path is not None:
            if model is not None:
                raise ValueError("Pass either model or checkpoint_path, not both.")

            self.model = load_transformer_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=self.device,
            )
        else:
            self.model = model or TransformerPolicyValue()

        self.model.to(self.device)
        self.model.eval()
        self.deterministic = deterministic
        self.temperature = temperature

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=False):
            raise ValueError("No legal moves available because the game is over.")

        encoded = encode_position(board)

        board_tokens = encoded["board_tokens"].unsqueeze(0).to(self.device)
        side_to_move = encoded["side_to_move"].unsqueeze(0).to(self.device)
        castling_rights = encoded["castling_rights"].unsqueeze(0).to(self.device)
        en_passant = encoded["en_passant"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(
                board_tokens=board_tokens,
                side_to_move=side_to_move,
                castling_rights=castling_rights,
                en_passant=en_passant,
            )

        logits = output.policy_logits.squeeze(0).detach().cpu()
        mask = legal_move_mask(board)

        masked_logits = logits.masked_fill(~mask, float("-inf"))

        if self.deterministic:
            move_id = int(torch.argmax(masked_logits).item())
        else:
            legal_probs = torch.softmax(masked_logits / self.temperature, dim=0)
            move_id = int(torch.multinomial(legal_probs, num_samples=1).item())

        move = id_to_move(move_id)

        if move not in board.legal_moves:
            raise ValueError(f"NeuralEngine produced illegal move after masking: {move}")

        return move