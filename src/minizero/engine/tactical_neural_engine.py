from __future__ import annotations

from pathlib import Path

import chess
import torch

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import id_to_move, legal_move_mask, move_to_id
from minizero.engine.neural_engine import NeuralEngine


PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class TacticalNeuralEngine(NeuralEngine):
    """Neural policy engine with cheap deterministic tactical guardrails.

    Move priority:
      1. Play an immediate checkmate if available.
      2. Avoid moves that allow opponent mate in 1, when possible.
      3. Capture the most valuable undefended opponent piece, when available.
      4. Fall back to the neural policy restricted to the safe move set.

    This is intentionally not a minimax/search engine. It only checks direct
    one-ply tactical facts and then delegates the general move choice to the
    trained neural policy.
    """

    name = "tactical_neural"

    def __init__(
        self,
        model=None,
        device: str | torch.device | None = None,
        deterministic: bool = True,
        temperature: float = 1.0,
        checkpoint_path: str | Path | None = None,
        use_mate_in_1: bool = True,
        avoid_opponent_mate_in_1: bool = True,
        capture_free_pieces: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            deterministic=deterministic,
            temperature=temperature,
            checkpoint_path=checkpoint_path,
        )
        self.use_mate_in_1 = use_mate_in_1
        self.avoid_opponent_mate_in_1 = avoid_opponent_mate_in_1
        self.capture_free_pieces = capture_free_pieces

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=False):
            raise ValueError("No legal moves available because the game is over.")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        if self.use_mate_in_1:
            mate_move = find_mate_in_1(board, legal_moves)
            if mate_move is not None:
                return mate_move

        candidate_moves = legal_moves
        if self.avoid_opponent_mate_in_1:
            safe_moves = [
                move for move in legal_moves
                if not allows_opponent_mate_in_1(board, move)
            ]
            if safe_moves:
                candidate_moves = safe_moves

        logits = self._policy_logits(board)

        if self.capture_free_pieces:
            free_capture = best_free_capture(board, candidate_moves, logits)
            if free_capture is not None:
                return free_capture

        return self._choose_neural_from_candidates(board, candidate_moves, logits)

    def _policy_logits(self, board: chess.Board) -> torch.Tensor:
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

        return output.policy_logits.squeeze(0).detach().cpu()

    def _choose_neural_from_candidates(
        self,
        board: chess.Board,
        candidate_moves: list[chess.Move],
        logits: torch.Tensor,
    ) -> chess.Move:
        mask = torch.zeros_like(legal_move_mask(board), dtype=torch.bool)
        for move in candidate_moves:
            mask[move_to_id(move)] = True

        masked_logits = logits.masked_fill(~mask, float("-inf"))

        if self.deterministic:
            move_id = int(torch.argmax(masked_logits).item())
        else:
            legal_probs = torch.softmax(masked_logits / self.temperature, dim=0)
            move_id = int(torch.multinomial(legal_probs, num_samples=1).item())

        move = id_to_move(move_id)

        if move not in candidate_moves:
            raise ValueError(f"TacticalNeuralEngine produced illegal/censored move: {move}")

        return move


def find_mate_in_1(
    board: chess.Board,
    legal_moves: list[chess.Move] | None = None,
) -> chess.Move | None:
    moves = legal_moves if legal_moves is not None else list(board.legal_moves)

    for move in moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move

    return None


def allows_opponent_mate_in_1(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    allows_mate = find_mate_in_1(board) is not None
    board.pop()
    return allows_mate


def best_free_capture(
    board: chess.Board,
    candidate_moves: list[chess.Move],
    logits: torch.Tensor | None = None,
) -> chess.Move | None:
    best_move: chess.Move | None = None
    best_key: tuple[int, float] | None = None

    for move in candidate_moves:
        if not board.is_capture(move):
            continue

        value = captured_piece_value(board, move)
        if value <= 0:
            continue

        target_square = captured_square(board, move)
        if board.attackers(not board.turn, target_square):
            continue

        logit_score = 0.0
        if logits is not None:
            logit_score = float(logits[move_to_id(move)].item())

        key = (value, logit_score)
        if best_key is None or key > best_key:
            best_key = key
            best_move = move

    return best_move


def captured_square(board: chess.Board, move: chess.Move) -> chess.Square:
    if board.is_en_passant(move):
        return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    return move.to_square


def captured_piece_value(board: chess.Board, move: chess.Move) -> int:
    square = captured_square(board, move)
    piece = board.piece_at(square)
    if piece is None:
        return 0
    return PIECE_VALUES[piece.piece_type]
