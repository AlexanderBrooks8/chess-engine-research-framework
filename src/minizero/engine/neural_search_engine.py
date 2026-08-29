from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import legal_move_mask, move_to_id
from minizero.engine.base import BaseEngine
from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.transformer_policy_value import TransformerPolicyValue


@dataclass(frozen=True)
class PositionEval:
    policy_logits: torch.Tensor
    policy_probs: torch.Tensor
    value_side_to_move: float


@dataclass(frozen=True)
class RootCandidate:
    move: chess.Move
    prior: float
    board_after_move: chess.Board
    value_score: float | None = None


class NeuralSearchEngine(BaseEngine):
    name = "neural_search"

    def __init__(
        self,
        model: TransformerPolicyValue | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        search_depth: int = 1,
        depth: int | None = None,
        root_top_k: int = 8,
        reply_top_k: int = 4,
        policy_weight: float = 0.25,
        value_weight: float = 1.0,
        policy_temperature: float = 1.0,
        use_eval_cache: bool = True,
        max_eval_cache_size: int = 100_000,
    ) -> None:
        if model is not None and checkpoint_path is not None:
            raise ValueError("Pass either model or checkpoint_path, not both.")

        if depth is not None:
            search_depth = depth

        if search_depth < 1:
            raise ValueError("search_depth must be >= 1.")
        if root_top_k <= 0:
            raise ValueError("root_top_k must be positive.")
        if reply_top_k <= 0:
            raise ValueError("reply_top_k must be positive.")
        if policy_temperature <= 0:
            raise ValueError("policy_temperature must be positive.")
        if max_eval_cache_size <= 0:
            raise ValueError("max_eval_cache_size must be positive.")

        self.device = torch.device(device or "cpu")

        if checkpoint_path is not None:
            self.model = load_transformer_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=self.device,
            )
        else:
            self.model = model or TransformerPolicyValue()

        self.model.to(self.device)
        self.model.eval()

        self.search_depth = search_depth
        self.root_top_k = root_top_k
        self.reply_top_k = reply_top_k
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.policy_temperature = policy_temperature

        self.use_eval_cache = use_eval_cache
        self.max_eval_cache_size = max_eval_cache_size
        self._eval_cache: OrderedDict[str, PositionEval] = OrderedDict()

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise ValueError("No legal moves available because the game is over.")

        root_turn = board.turn
        root_eval = self._evaluate_board(board)

        root_moves = self._top_policy_moves(
            board=board,
            policy_probs=root_eval.policy_probs,
            limit=self.root_top_k,
        )

        if not root_moves:
            raise ValueError("No legal moves available.")

        immediate_mates = self._immediate_checkmate_moves(board, root_moves)
        if immediate_mates:
            return self._best_prior_move(immediate_mates, root_eval.policy_probs)

        candidates: list[RootCandidate] = []

        for move in root_moves:
            child = board.copy(stack=False)
            child.push(move)

            prior = self._move_probability(root_eval.policy_probs, move)

            if child.is_game_over(claim_draw=True):
                value_score = self._terminal_value_for_root(child, root_turn)
                candidates.append(
                    RootCandidate(
                        move=move,
                        prior=prior,
                        board_after_move=child,
                        value_score=value_score,
                    )
                )
            else:
                candidates.append(
                    RootCandidate(
                        move=move,
                        prior=prior,
                        board_after_move=child,
                        value_score=None,
                    )
                )

        if self.search_depth == 1:
            candidates = self._score_depth_one(candidates, root_turn)
        else:
            candidates = self._score_depth_two(candidates, root_turn)

        best_move = candidates[0].move
        best_total_score = float("-inf")

        for candidate in candidates:
            if candidate.value_score is None:
                continue

            total_score = (
                self.value_weight * candidate.value_score
                + self.policy_weight * candidate.prior
            )

            if total_score > best_total_score:
                best_total_score = total_score
                best_move = candidate.move

        if best_move not in board.legal_moves:
            raise ValueError(f"NeuralSearchEngine produced illegal move: {best_move}")

        return best_move

    def _score_depth_one(
        self,
        candidates: list[RootCandidate],
        root_turn: chess.Color,
    ) -> list[RootCandidate]:
        boards_to_eval: list[chess.Board] = []
        candidate_indices: list[int] = []

        for idx, candidate in enumerate(candidates):
            if candidate.value_score is None:
                boards_to_eval.append(candidate.board_after_move)
                candidate_indices.append(idx)

        evals = self._evaluate_boards(boards_to_eval) if boards_to_eval else []

        updated = list(candidates)

        for candidate_idx, pos_eval in zip(candidate_indices, evals):
            board_after_move = candidates[candidate_idx].board_after_move
            value_score = self._model_value_for_root(
                value_side_to_move=pos_eval.value_side_to_move,
                board_turn=board_after_move.turn,
                root_turn=root_turn,
            )

            updated[candidate_idx] = RootCandidate(
                move=candidates[candidate_idx].move,
                prior=candidates[candidate_idx].prior,
                board_after_move=board_after_move,
                value_score=value_score,
            )

        return updated

    def _score_depth_two(
        self,
        candidates: list[RootCandidate],
        root_turn: chess.Color,
    ) -> list[RootCandidate]:
        unresolved_indices: list[int] = []
        unresolved_boards: list[chess.Board] = []

        for idx, candidate in enumerate(candidates):
            if candidate.value_score is None:
                unresolved_indices.append(idx)
                unresolved_boards.append(candidate.board_after_move)

        child_evals = self._evaluate_boards(unresolved_boards) if unresolved_boards else []

        reply_boards: list[chess.Board] = []
        reply_owner_indices: list[int] = []
        reply_terminal_scores: dict[int, list[float]] = {
            idx: [] for idx in range(len(candidates))
        }

        for candidate_idx, child_eval in zip(unresolved_indices, child_evals):
            child_board = candidates[candidate_idx].board_after_move

            replies = self._top_policy_moves(
                board=child_board,
                policy_probs=child_eval.policy_probs,
                limit=self.reply_top_k,
            )

            if not replies:
                reply_terminal_scores[candidate_idx].append(
                    self._model_value_for_root(
                        value_side_to_move=child_eval.value_side_to_move,
                        board_turn=child_board.turn,
                        root_turn=root_turn,
                    )
                )
                continue

            for reply in replies:
                reply_board = child_board.copy(stack=False)
                reply_board.push(reply)

                if reply_board.is_game_over(claim_draw=True):
                    reply_terminal_scores[candidate_idx].append(
                        self._terminal_value_for_root(reply_board, root_turn)
                    )
                else:
                    reply_owner_indices.append(candidate_idx)
                    reply_boards.append(reply_board)

        reply_evals = self._evaluate_boards(reply_boards) if reply_boards else []

        for owner_idx, reply_board, reply_eval in zip(
            reply_owner_indices,
            reply_boards,
            reply_evals,
        ):
            reply_terminal_scores[owner_idx].append(
                self._model_value_for_root(
                    value_side_to_move=reply_eval.value_side_to_move,
                    board_turn=reply_board.turn,
                    root_turn=root_turn,
                )
            )

        updated = list(candidates)

        for idx, candidate in enumerate(candidates):
            if candidate.value_score is not None:
                continue

            reply_scores = reply_terminal_scores[idx]

            if not reply_scores:
                value_score = 0.0
            else:
                value_score = min(reply_scores)

            updated[idx] = RootCandidate(
                move=candidate.move,
                prior=candidate.prior,
                board_after_move=candidate.board_after_move,
                value_score=value_score,
            )

        return updated

    def _evaluate_board(self, board: chess.Board) -> PositionEval:
        return self._evaluate_boards([board])[0]

    def _evaluate_boards(self, boards: list[chess.Board]) -> list[PositionEval]:
        if not boards:
            return []

        if not self.use_eval_cache:
            return self._evaluate_boards_uncached(boards)

        results: list[PositionEval | None] = [None] * len(boards)
        missed_boards: list[chess.Board] = []
        missed_indices: list[int] = []

        for idx, board in enumerate(boards):
            key = self._cache_key(board)
            cached = self._eval_cache.get(key)

            if cached is not None:
                self._eval_cache.move_to_end(key)
                results[idx] = cached
            else:
                missed_boards.append(board)
                missed_indices.append(idx)

        if missed_boards:
            missed_results = self._evaluate_boards_uncached(missed_boards)

            for idx, board, pos_eval in zip(missed_indices, missed_boards, missed_results):
                key = self._cache_key(board)
                self._eval_cache[key] = pos_eval
                self._eval_cache.move_to_end(key)
                results[idx] = pos_eval

                while len(self._eval_cache) > self.max_eval_cache_size:
                    self._eval_cache.popitem(last=False)

        return [result for result in results if result is not None]

    def _evaluate_boards_uncached(self, boards: list[chess.Board]) -> list[PositionEval]:
        if not boards:
            return []

        encoded_positions = [encode_position(board) for board in boards]

        board_tokens = torch.stack(
            [encoded["board_tokens"] for encoded in encoded_positions],
            dim=0,
        ).to(self.device)

        side_to_move = torch.stack(
            [encoded["side_to_move"] for encoded in encoded_positions],
            dim=0,
        ).to(self.device)

        castling_rights = torch.stack(
            [encoded["castling_rights"] for encoded in encoded_positions],
            dim=0,
        ).to(self.device)

        en_passant = torch.stack(
            [encoded["en_passant"] for encoded in encoded_positions],
            dim=0,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(
                board_tokens=board_tokens,
                side_to_move=side_to_move,
                castling_rights=castling_rights,
                en_passant=en_passant,
            )

        policy_logits = output.policy_logits.detach().cpu()
        values = self._extract_values(output).detach().float().cpu().view(-1)

        results: list[PositionEval] = []

        for idx, board in enumerate(boards):
            logits = policy_logits[idx]
            probs = self._masked_policy_probs(logits, board)
            results.append(
                PositionEval(
                    policy_logits=logits,
                    policy_probs=probs,
                    value_side_to_move=float(values[idx].item()),
                )
            )

        return results

    def _extract_values(self, output: object) -> torch.Tensor:
        for attr in ("value", "value_logits", "values", "value_pred", "pred_value"):
            if hasattr(output, attr):
                value = getattr(output, attr)
                if isinstance(value, torch.Tensor):
                    return value

        raise AttributeError(
            "Model output must expose a tensor value field. Tried: "
            "value, value_logits, values, value_pred, pred_value."
        )

    def _masked_policy_probs(
        self,
        logits: torch.Tensor,
        board: chess.Board,
    ) -> torch.Tensor:
        mask = legal_move_mask(board)

        if not bool(mask.any().item()):
            return torch.zeros_like(logits, dtype=torch.float32)

        masked_logits = logits.float().masked_fill(~mask, float("-inf"))
        return torch.softmax(masked_logits / self.policy_temperature, dim=0)

    def _top_policy_moves(
        self,
        board: chess.Board,
        policy_probs: torch.Tensor,
        limit: int,
    ) -> list[chess.Move]:
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return []

        scored_moves = [
            (self._move_probability(policy_probs, move), move)
            for move in legal_moves
        ]

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored_moves[:limit]]

    def _move_probability(
        self,
        policy_probs: torch.Tensor,
        move: chess.Move,
    ) -> float:
        return float(policy_probs[move_to_id(move)].item())

    def _model_value_for_root(
        self,
        value_side_to_move: float,
        board_turn: chess.Color,
        root_turn: chess.Color,
    ) -> float:
        del board_turn

        if root_turn == chess.WHITE:
            return value_side_to_move

        return -value_side_to_move

    def _terminal_value_for_root(
        self,
        board: chess.Board,
        root_turn: chess.Color,
    ) -> float:
        outcome = board.outcome(claim_draw=True)

        if outcome is None or outcome.winner is None:
            return 0.0

        return 1.0 if outcome.winner == root_turn else -1.0

    def _immediate_checkmate_moves(
        self,
        board: chess.Board,
        candidate_moves: list[chess.Move],
    ) -> list[chess.Move]:
        mates: list[chess.Move] = []

        for move in candidate_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()

            if is_mate:
                mates.append(move)

        return mates

    def _best_prior_move(
        self,
        moves: list[chess.Move],
        policy_probs: torch.Tensor,
    ) -> chess.Move:
        best_move = moves[0]
        best_prior = float("-inf")

        for move in moves:
            prior = self._move_probability(policy_probs, move)

            if prior > best_prior:
                best_prior = prior
                best_move = move

        return best_move

    def _cache_key(self, board: chess.Board) -> str:
        return board.fen()