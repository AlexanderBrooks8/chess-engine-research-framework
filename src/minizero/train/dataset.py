from __future__ import annotations

from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset

from minizero.chess.move_vocab import legal_move_mask
from minizero.selfplay.game_record import TrainingExample, load_game_record


class ReplayDataset(Dataset):
    def __init__(self, paths: list[str | Path]) -> None:
        self.examples: list[TrainingExample] = []

        for path in paths:
            record = load_game_record(path)

            if record.result is None:
                raise ValueError(f"Game record has no result: {path}")

            self.examples.extend(record.finalize(record.result))

        if not self.examples:
            raise ValueError("ReplayDataset received no training examples.")

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.pt") -> ReplayDataset:
        directory = Path(directory)
        paths = sorted(directory.glob(pattern))

        if not paths:
            raise ValueError(f"No replay files found in {directory} with pattern {pattern}.")

        return cls(paths)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


def collate_training_examples(examples: list[TrainingExample]) -> dict[str, torch.Tensor]:
    legal_masks = [
        legal_move_mask(chess.Board(example.fen))
        for example in examples
    ]

    return {
        "board_tokens": torch.stack([example.board_tokens for example in examples]),
        "side_to_move": torch.stack([example.side_to_move for example in examples]),
        "castling_rights": torch.stack([example.castling_rights for example in examples]),
        "en_passant": torch.stack([example.en_passant for example in examples]),
        "target_policy": torch.stack([example.target_policy for example in examples]),
        "target_value": torch.stack([example.target_value for example in examples]),
        "legal_mask": torch.stack(legal_masks),
    }


def split_batch(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    model_inputs = {
        "board_tokens": batch["board_tokens"],
        "side_to_move": batch["side_to_move"],
        "castling_rights": batch["castling_rights"],
        "en_passant": batch["en_passant"],
    }

    target_policy = batch["target_policy"]
    target_value = batch["target_value"]
    legal_mask = batch["legal_mask"]

    return model_inputs, target_policy, target_value, legal_mask