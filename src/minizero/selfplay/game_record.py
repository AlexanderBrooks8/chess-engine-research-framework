from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import torch

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import VOCAB_SIZE, move_to_id


@dataclass(frozen=True)
class GameStep:
    fen: str
    move_uci: str
    move_id: int
    player: bool
    policy_target: torch.Tensor | None = None
    value_target: float | None = None


@dataclass(frozen=True)
class TrainingExample:
    fen: str
    move_uci: str
    move_id: int
    player: bool
    board_tokens: torch.Tensor
    side_to_move: torch.Tensor
    castling_rights: torch.Tensor
    en_passant: torch.Tensor
    target_policy: torch.Tensor
    target_value: torch.Tensor


class GameRecord:
    def __init__(self) -> None:
        self.steps: list[GameStep] = []
        self.result: str | None = None

    def add_position(
        self,
        board: chess.Board,
        move: chess.Move,
        policy_target: torch.Tensor | None = None,
        value_target: float | None = None,
    ) -> None:
        if move not in board.legal_moves:
            raise ValueError(f"Cannot record illegal move {move.uci()} for board {board.fen()}.")

        if policy_target is not None:
            if policy_target.shape != (VOCAB_SIZE,):
                raise ValueError(
                    f"policy_target must have shape ({VOCAB_SIZE},), got {tuple(policy_target.shape)}."
                )

            policy_target = policy_target.detach().cpu().to(dtype=torch.float32).clone()

        if value_target is not None:
            value_target = float(value_target)

            if value_target < -1.0 or value_target > 1.0:
                raise ValueError(f"value_target must be in [-1, 1], got {value_target}.")

        self.steps.append(
            GameStep(
                fen=board.fen(),
                move_uci=move.uci(),
                move_id=move_to_id(move),
                player=board.turn,
                policy_target=policy_target,
                value_target=value_target,
            )
        )

    def finalize(self, result: str) -> list[TrainingExample]:
        self.result = result

        examples: list[TrainingExample] = []

        for step in self.steps:
            board = chess.Board(step.fen)
            encoded = encode_position(board)

            if step.policy_target is not None:
                target_policy = step.policy_target.clone().to(dtype=torch.float32)
            else:
                target_policy = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
                target_policy[step.move_id] = 1.0

            if step.value_target is not None:
                target_value_float = step.value_target
            else:
                target_value_float = result_to_value(result=result, player=step.player)

            target_value = torch.tensor(target_value_float, dtype=torch.float32)

            examples.append(
                TrainingExample(
                    fen=step.fen,
                    move_uci=step.move_uci,
                    move_id=step.move_id,
                    player=step.player,
                    board_tokens=encoded["board_tokens"],
                    side_to_move=encoded["side_to_move"],
                    castling_rights=encoded["castling_rights"],
                    en_passant=encoded["en_passant"],
                    target_policy=target_policy,
                    target_value=target_value,
                )
            )

        return examples

    def to_dict(self) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []

        for step in self.steps:
            step_data: dict[str, Any] = {
                "fen": step.fen,
                "move_uci": step.move_uci,
                "move_id": step.move_id,
                "player": step.player,
            }

            if step.policy_target is not None:
                step_data["policy_target"] = step.policy_target

            if step.value_target is not None:
                step_data["value_target"] = step.value_target

            steps.append(step_data)

        return {
            "steps": steps,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        record = cls()
        record.steps = [
            GameStep(
                fen=step["fen"],
                move_uci=step["move_uci"],
                move_id=int(step["move_id"]),
                player=bool(step["player"]),
                policy_target=(
                    step["policy_target"].detach().cpu().to(dtype=torch.float32).clone()
                    if "policy_target" in step
                    else None
                ),
                value_target=(
                    float(step["value_target"])
                    if "value_target" in step
                    else None
                ),
            )
            for step in data["steps"]
        ]
        record.result = data["result"]
        return record


def result_to_value(result: str, player: bool) -> float:
    if result == "1-0":
        return 1.0 if player == chess.WHITE else -1.0

    if result == "0-1":
        return -1.0 if player == chess.WHITE else 1.0

    if result == "1/2-1/2":
        return 0.0

    raise ValueError(f"Unsupported game result: {result}")


def save_game_record(path: str | Path, record: GameRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record.to_dict(), path)


def load_game_record(path: str | Path) -> GameRecord:
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return GameRecord.from_dict(data)
