from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import chess

from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.transformer_policy_value import TransformerPolicyValue
from minizero.search.mcts import run_mcts
from minizero.selfplay.game_record import GameRecord, save_game_record

@dataclass(frozen=True)
class MCTSSelfPlayWorkerConfig:
    output_dir: str
    max_plies: int
    num_simulations: int
    c_puct: float
    temperature: float
    root_dirichlet_alpha: float | None
    root_exploration_fraction: float


_WORKER_MODEL = None
_WORKER_CONFIG: MCTSSelfPlayWorkerConfig | None = None


def _init_mcts_selfplay_worker(
    checkpoint_path: str | None,
    config: MCTSSelfPlayWorkerConfig,
) -> None:
    global _WORKER_MODEL
    global _WORKER_CONFIG

    _WORKER_MODEL = load_model_for_selfplay(checkpoint_path)
    _WORKER_CONFIG = config


def _generate_one_mcts_selfplay_game(game_index: int) -> str:
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model was not initialized.")

    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker config was not initialized.")

    result = play_mcts_selfplay_game(
        model=_WORKER_MODEL,
        max_plies=_WORKER_CONFIG.max_plies,
        num_simulations=_WORKER_CONFIG.num_simulations,
        c_puct=_WORKER_CONFIG.c_puct,
        temperature=_WORKER_CONFIG.temperature,
        root_dirichlet_alpha=_WORKER_CONFIG.root_dirichlet_alpha,
        root_exploration_fraction=_WORKER_CONFIG.root_exploration_fraction,
    )

    path = Path(_WORKER_CONFIG.output_dir) / f"mcts_game_{game_index:06d}.pt"
    save_game_record(path, result.record)

    return str(path)

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def material_score(board: chess.Board) -> float:
    score = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            continue

        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value

    return score


def adjudicate_result_by_material(
    board: chess.Board,
    threshold: float = 1.0,
) -> str:
    score = material_score(board)

    if score >= threshold:
        return "1-0"

    if score <= -threshold:
        return "0-1"

    return "1/2-1/2"

@dataclass(frozen=True)
class MCTSSelfPlayGameResult:
    record: GameRecord
    result: str
    plies: int
    termination: str


def play_mcts_selfplay_game(
    model: TransformerPolicyValue,
    max_plies: int = 256,
    num_simulations: int = 32,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    root_dirichlet_alpha: float | None = 0.3,
    root_exploration_fraction: float = 0.25,
) -> MCTSSelfPlayGameResult:
    board = chess.Board()
    record = GameRecord()
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        mcts_result = run_mcts(
            board=board,
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
        )

        move = mcts_result.move

        if move not in board.legal_moves:
            raise ValueError(f"MCTS produced illegal move: {move}")

        record.add_position(
            board=board,
            move=move,
            policy_target=mcts_result.policy_target,
        )

        board.push(move)
        plies += 1

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = board.result(claim_draw=True)
        termination = str(outcome.termination) if outcome is not None else "unknown"
    else:
        result = adjudicate_result_by_material(board)
        termination = "max_plies_adjudicated_material"

    record.finalize(result)

    return MCTSSelfPlayGameResult(
        record=record,
        result=result,
        plies=plies,
        termination=termination,
    )


def load_model_for_selfplay(
    checkpoint_path: str | Path | None,
) -> TransformerPolicyValue:
    if checkpoint_path is None:
        return TransformerPolicyValue(
            d_model=64,
            n_layers=1,
            n_heads=4,
            ff_dim=128,
            dropout=0.0,
        )

    return load_transformer_from_checkpoint(checkpoint_path)


def generate_mcts_selfplay_games(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    games: int = 10,
    max_plies: int = 256,
    num_simulations: int = 32,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    root_dirichlet_alpha: float | None = 0.3,
    root_exploration_fraction: float = 0.25,
    num_workers: int = 1,
) -> list[Path]:
    if games <= 0:
        raise ValueError("games must be positive.")

    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_str = str(checkpoint_path) if checkpoint_path is not None else None

    if num_workers == 1:
        model = load_model_for_selfplay(checkpoint_path)
        saved_paths: list[Path] = []

        for game_index in range(games):
            result = play_mcts_selfplay_game(
                model=model,
                max_plies=max_plies,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature=temperature,
                root_dirichlet_alpha=root_dirichlet_alpha,
                root_exploration_fraction=root_exploration_fraction,
            )

            path = output_dir / f"mcts_game_{game_index:06d}.pt"
            save_game_record(path, result.record)
            saved_paths.append(path)

        return saved_paths

    config = MCTSSelfPlayWorkerConfig(
        output_dir=str(output_dir),
        max_plies=max_plies,
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=temperature,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
    )

    worker_count = min(num_workers, games)

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_mcts_selfplay_worker,
        initargs=(checkpoint_str, config),
    ) as executor:
        saved_paths = list(
            executor.map(
                _generate_one_mcts_selfplay_game,
                range(games),
            )
        )

    return [Path(path) for path in saved_paths]