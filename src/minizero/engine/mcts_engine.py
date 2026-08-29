from __future__ import annotations

from pathlib import Path

import chess
import torch

from minizero.engine.base import BaseEngine
from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.transformer_policy_value import TransformerPolicyValue
from minizero.search.mcts import MCTSEvaluationCache, run_mcts
from minizero.search.tactical_veto import choose_tactical_veto_move


class MCTSEngine(BaseEngine):
    name = "mcts"

    def __init__(
        self,
        model: TransformerPolicyValue | None = None,
        checkpoint_path: str | Path | None = None,
        num_simulations: int | None = 32,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        root_dirichlet_alpha: float | None = None,
        root_exploration_fraction: float = 0.25,
        time_limit_ms: float | None = None,
        leaf_batch_size: int = 1,
        max_children: int | None = None,
        tactical_veto: bool = False,
        veto_top_n: int = 8,
        veto_depth: int = 2,
        veto_threshold_pawns: float = 2.0,
        veto_mobility_weight: float = 0.01,
        device: str | torch.device | None = None,
        use_eval_cache: bool = True,
        eval_cache_size: int = 100_000,
    ) -> None:
        if num_simulations is not None and num_simulations <= 0:
            raise ValueError("num_simulations must be positive when provided.")

        if time_limit_ms is not None and time_limit_ms <= 0:
            raise ValueError("time_limit_ms must be positive when provided.")

        if num_simulations is None and time_limit_ms is None:
            raise ValueError("Pass num_simulations and/or time_limit_ms.")

        if leaf_batch_size <= 0:
            raise ValueError("leaf_batch_size must be positive.")

        if max_children is not None and max_children <= 0:
            raise ValueError("max_children must be positive when provided.")

        if veto_top_n <= 0:
            raise ValueError("veto_top_n must be positive.")

        if veto_depth < 1:
            raise ValueError("veto_depth must be at least 1.")

        if veto_threshold_pawns < 0:
            raise ValueError("veto_threshold_pawns must be non-negative.")

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

        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.time_limit_ms = time_limit_ms
        self.leaf_batch_size = leaf_batch_size
        self.max_children = max_children
        self.tactical_veto = tactical_veto
        self.veto_top_n = veto_top_n
        self.veto_depth = veto_depth
        self.veto_threshold_pawns = veto_threshold_pawns
        self.veto_mobility_weight = veto_mobility_weight
        self.eval_cache = (
            MCTSEvaluationCache(max_size=eval_cache_size)
            if use_eval_cache
            else None
        )

        self.last_simulations_run = 0
        self.last_elapsed_seconds = 0.0
        self.last_model_evaluations = 0
        self.last_model_batches = 0
        self.last_avg_model_batch_size = 0.0
        self.last_max_model_batch_size = 0
        self.last_cache_hits = 0
        self.last_cache_misses = 0

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=False):
            raise ValueError("No legal moves available because the game is over.")

        result = run_mcts(
            board=board,
            model=self.model,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            temperature=self.temperature,
            root_dirichlet_alpha=self.root_dirichlet_alpha,
            root_exploration_fraction=self.root_exploration_fraction,
            time_limit_s=(
                None
                if self.time_limit_ms is None
                else self.time_limit_ms / 1000.0
            ),
            eval_cache=self.eval_cache,
            leaf_batch_size=self.leaf_batch_size,
            max_children=self.max_children,
        )

        self.last_simulations_run = result.simulations_run
        self.last_elapsed_seconds = result.elapsed_seconds
        self.last_model_evaluations = result.model_evaluations
        self.last_model_batches = result.model_batches
        self.last_avg_model_batch_size = result.avg_model_batch_size
        self.last_max_model_batch_size = result.max_model_batch_size
        self.last_cache_hits = result.cache_hits
        self.last_cache_misses = result.cache_misses

        if not self.tactical_veto:
            return result.move

        veto_decision = choose_tactical_veto_move(
            board=board,
            preferred_move=result.move,
            visit_counts=result.visit_counts,
            top_n=self.veto_top_n,
            depth=self.veto_depth,
            threshold_pawns=self.veto_threshold_pawns,
            mobility_weight=self.veto_mobility_weight,
        )

        return veto_decision.selected_move
