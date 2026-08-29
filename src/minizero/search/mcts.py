import time
from dataclasses import dataclass
from typing import Iterable

import chess
import torch
import torch.nn.functional as F

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import VOCAB_SIZE, move_to_id
from minizero.models.transformer_policy_value import TransformerPolicyValue
from minizero.search.node import MCTSNode
from minizero.search.puct import puct_score


@dataclass(frozen=True)
class MCTSEvaluation:
    value: float
    move_priors: dict[chess.Move, float]


@dataclass
class MCTSStats:
    simulations_run: int = 0
    model_evaluations: int = 0
    model_batches: int = 0
    model_batch_positions: int = 0
    max_model_batch_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    elapsed_seconds: float = 0.0

    @property
    def avg_model_batch_size(self) -> float:
        if self.model_batches <= 0:
            return 0.0
        return self.model_batch_positions / self.model_batches


@dataclass(frozen=True)
class MCTSResult:
    move: chess.Move
    visit_counts: dict[chess.Move, int]
    policy_target: torch.Tensor
    root_value: float
    simulations_run: int = 0
    elapsed_seconds: float = 0.0
    model_evaluations: int = 0
    model_batches: int = 0
    avg_model_batch_size: float = 0.0
    max_model_batch_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(frozen=True)
class PendingLeaf:
    node: MCTSNode
    board: chess.Board
    search_path: list[MCTSNode]


class MCTSEvaluationCache:
    def __init__(self, max_size: int = 100_000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive.")

        self.max_size = max_size
        self._cache: dict[str, MCTSEvaluation] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> MCTSEvaluation | None:
        cached = self._cache.get(key)

        if cached is None:
            self.misses += 1
            return None

        self.hits += 1
        return cached

    def put(self, key: str, evaluation: MCTSEvaluation) -> None:
        if len(self._cache) >= self.max_size:
            # Predictable FIFO-style eviction keeps cache behavior deterministic.
            self._cache.pop(next(iter(self._cache)))

        self._cache[key] = evaluation

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)


def model_device(model: TransformerPolicyValue) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def position_key(board: chess.Board) -> str:
    # The model only sees board pieces, side to move, castling rights, and en-passant.
    # Halfmove/fullmove clocks are intentionally excluded from the cache key.
    return " ".join(
        [
            board.board_fen(),
            "w" if board.turn == chess.WHITE else "b",
            chess.Board.castling_xfen(board),
            chess.square_name(board.ep_square) if board.ep_square is not None else "-",
        ]
    )


def search_terminal_value(board: chess.Board) -> float | None:
    if board.is_checkmate():
        # Side to move is checkmated, so from side-to-move perspective this is -1.
        return -1.0

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    return None


def encode_batch(
    board: chess.Board,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    encoded = encode_position(board)

    batch = {
        "board_tokens": encoded["board_tokens"].unsqueeze(0),
        "side_to_move": encoded["side_to_move"].unsqueeze(0),
        "castling_rights": encoded["castling_rights"].unsqueeze(0),
        "en_passant": encoded["en_passant"].unsqueeze(0),
    }

    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}

    return batch


def encode_boards(
    boards: list[chess.Board],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded_positions = [encode_position(board) for board in boards]

    batch = {
        "board_tokens": torch.stack([encoded["board_tokens"] for encoded in encoded_positions]),
        "side_to_move": torch.stack([encoded["side_to_move"] for encoded in encoded_positions]),
        "castling_rights": torch.stack([encoded["castling_rights"] for encoded in encoded_positions]),
        "en_passant": torch.stack([encoded["en_passant"] for encoded in encoded_positions]),
    }

    return {key: value.to(device) for key, value in batch.items()}


def prune_move_priors(
    move_priors: dict[chess.Move, float],
    max_children: int | None,
) -> dict[chess.Move, float]:
    if max_children is None or max_children <= 0 or len(move_priors) <= max_children:
        return move_priors

    top_items = sorted(
        move_priors.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max_children]

    total = sum(prior for _, prior in top_items)

    if total <= 0.0:
        uniform = 1.0 / len(top_items)
        return {move: uniform for move, _ in top_items}

    return {move: prior / total for move, prior in top_items}


def evaluation_from_output(
    board: chess.Board,
    logits: torch.Tensor,
    value: float,
) -> MCTSEvaluation:
    legal_moves = list(board.legal_moves)
    legal_move_ids = [move_to_id(move) for move in legal_moves]

    legal_logits = logits[legal_move_ids]
    legal_probs = F.softmax(legal_logits, dim=-1)

    move_priors = {
        move: float(prob.item())
        for move, prob in zip(legal_moves, legal_probs)
    }

    return MCTSEvaluation(
        value=value,
        move_priors=move_priors,
    )


def evaluate_positions(
    boards: list[chess.Board],
    model: TransformerPolicyValue,
    cache: MCTSEvaluationCache | None = None,
    stats: MCTSStats | None = None,
) -> list[MCTSEvaluation]:
    if not boards:
        return []

    keys = [position_key(board) for board in boards]
    results: list[MCTSEvaluation | None] = [None] * len(boards)
    miss_indices: list[int] = []

    for idx, key in enumerate(keys):
        cached = cache.get(key) if cache is not None else None

        if cached is not None:
            results[idx] = cached
            if stats is not None:
                stats.cache_hits += 1
        else:
            miss_indices.append(idx)
            if cache is not None and stats is not None:
                stats.cache_misses += 1

    if miss_indices:
        device = model_device(model)
        miss_boards = [boards[idx] for idx in miss_indices]

        with torch.inference_mode():
            output = model(**encode_boards(miss_boards, device=device))

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        logits_batch = output.policy_logits.detach().cpu()
        values = output.value.detach().cpu().view(-1).tolist()

        for local_idx, board_idx in enumerate(miss_indices):
            evaluation = evaluation_from_output(
                board=boards[board_idx],
                logits=logits_batch[local_idx],
                value=float(values[local_idx]),
            )
            results[board_idx] = evaluation

            if cache is not None:
                cache.put(keys[board_idx], evaluation)

        if stats is not None:
            batch_size = len(miss_indices)
            stats.model_evaluations += batch_size
            stats.model_batches += 1
            stats.model_batch_positions += batch_size
            stats.max_model_batch_size = max(stats.max_model_batch_size, batch_size)

    return [evaluation for evaluation in results if evaluation is not None]


def evaluate_position(
    board: chess.Board,
    model: TransformerPolicyValue,
    cache: MCTSEvaluationCache | None = None,
    stats: MCTSStats | None = None,
) -> MCTSEvaluation:
    return evaluate_positions(
        boards=[board],
        model=model,
        cache=cache,
        stats=stats,
    )[0]


def evaluate_and_expand(
    node: MCTSNode,
    board: chess.Board,
    model: TransformerPolicyValue,
    cache: MCTSEvaluationCache | None = None,
    stats: MCTSStats | None = None,
    max_children: int | None = None,
) -> float:
    evaluation = evaluate_position(
        board=board,
        model=model,
        cache=cache,
        stats=stats,
    )

    node.expand(prune_move_priors(evaluation.move_priors, max_children=max_children))

    return evaluation.value


def add_root_dirichlet_noise(
    root: MCTSNode,
    alpha: float,
    exploration_fraction: float,
) -> None:
    if not root.children:
        return

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if not 0.0 <= exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be between 0 and 1.")

    moves = list(root.children.keys())

    noise = torch.distributions.Dirichlet(
        torch.full((len(moves),), alpha, dtype=torch.float32)
    ).sample()

    for move, noise_value in zip(moves, noise):
        child = root.children[move]
        child.prior = (
            (1.0 - exploration_fraction) * child.prior
            + exploration_fraction * float(noise_value.item())
        )


def select_child(
    node: MCTSNode,
    c_puct: float,
) -> tuple[chess.Move, MCTSNode]:
    if not node.children:
        raise ValueError("Cannot select child from an unexpanded node.")

    return max(
        node.children.items(),
        key=lambda item: puct_score(parent=node, child=item[1], c_puct=c_puct),
    )


def reserve_visit(search_path: Iterable[MCTSNode]) -> None:
    # For batched leaf selection, reserve each selected path before model inference.
    # This acts like a zero-value virtual visit: it discourages selecting the same leaf
    # repeatedly while preserving one visit per completed simulation.
    for node in search_path:
        node.visit_count += 1


def backup_value_only(
    search_path: list[MCTSNode],
    value: float,
) -> None:
    for node in reversed(search_path):
        node.value_sum += value

        # Alternate perspective each ply.
        value = -value


def backup(
    search_path: list[MCTSNode],
    value: float,
) -> None:
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value

        # Alternate perspective each ply.
        value = -value


def make_policy_target(visit_counts: dict[chess.Move, int]) -> torch.Tensor:
    policy = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    total_visits = sum(visit_counts.values())

    if total_visits <= 0:
        raise ValueError("Cannot make policy target with zero total visits.")

    for move, count in visit_counts.items():
        policy[move_to_id(move)] = count / total_visits

    return policy


def select_move_from_visits(
    visit_counts: dict[chess.Move, int],
    temperature: float,
) -> chess.Move:
    if temperature <= 0:
        return max(visit_counts.items(), key=lambda item: item[1])[0]

    moves = list(visit_counts.keys())
    counts = torch.tensor([visit_counts[move] for move in moves], dtype=torch.float32)

    if temperature != 1.0:
        counts = counts.pow(1.0 / temperature)

    probs = counts / counts.sum()
    selected_idx = int(torch.multinomial(probs, num_samples=1).item())

    return moves[selected_idx]


def _should_continue_search(
    simulations_run: int,
    num_simulations: int | None,
    deadline: float | None,
) -> bool:
    if num_simulations is not None and simulations_run >= num_simulations:
        return False

    if deadline is not None and time.perf_counter() >= deadline:
        return False

    return True


def run_single_leaf_simulation(
    root: MCTSNode,
    board: chess.Board,
    model: TransformerPolicyValue,
    c_puct: float,
    eval_cache: MCTSEvaluationCache | None,
    stats: MCTSStats,
    max_children: int | None,
) -> None:
    node = root
    sim_board = board.copy(stack=False)
    search_path = [node]

    while node.expanded:
        move, node = select_child(node, c_puct=c_puct)
        sim_board.push(move)
        search_path.append(node)

    terminal = search_terminal_value(sim_board)

    if terminal is not None:
        value = terminal
    else:
        value = evaluate_and_expand(
            node=node,
            board=sim_board,
            model=model,
            cache=eval_cache,
            stats=stats,
            max_children=max_children,
        )

    backup(search_path, value)
    stats.simulations_run += 1


def run_batched_leaf_simulations(
    root: MCTSNode,
    board: chess.Board,
    model: TransformerPolicyValue,
    c_puct: float,
    eval_cache: MCTSEvaluationCache | None,
    stats: MCTSStats,
    max_children: int | None,
    leaf_batch_size: int,
    num_simulations: int | None,
    deadline: float | None,
) -> None:
    pending: list[PendingLeaf] = []

    while len(pending) < leaf_batch_size and _should_continue_search(
        simulations_run=stats.simulations_run + len(pending),
        num_simulations=num_simulations,
        deadline=deadline,
    ):
        node = root
        sim_board = board.copy(stack=False)
        search_path = [node]

        while node.expanded:
            move, node = select_child(node, c_puct=c_puct)
            sim_board.push(move)
            search_path.append(node)

        reserve_visit(search_path)
        terminal = search_terminal_value(sim_board)

        if terminal is not None:
            backup_value_only(search_path, terminal)
            stats.simulations_run += 1
        else:
            pending.append(
                PendingLeaf(
                    node=node,
                    board=sim_board,
                    search_path=search_path,
                )
            )

    if not pending:
        return

    evaluations = evaluate_positions(
        boards=[leaf.board for leaf in pending],
        model=model,
        cache=eval_cache,
        stats=stats,
    )

    for leaf, evaluation in zip(pending, evaluations):
        leaf.node.expand(prune_move_priors(evaluation.move_priors, max_children=max_children))
        backup_value_only(leaf.search_path, evaluation.value)
        stats.simulations_run += 1


def run_mcts(
    board: chess.Board,
    model: TransformerPolicyValue,
    num_simulations: int | None = 32,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    root_dirichlet_alpha: float | None = None,
    root_exploration_fraction: float = 0.25,
    time_limit_s: float | None = None,
    eval_cache: MCTSEvaluationCache | None = None,
    leaf_batch_size: int = 1,
    max_children: int | None = None,
) -> MCTSResult:
    if num_simulations is not None and num_simulations <= 0:
        raise ValueError("num_simulations must be positive when provided.")

    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive when provided.")

    if num_simulations is None and time_limit_s is None:
        raise ValueError("Pass num_simulations and/or time_limit_s.")

    if leaf_batch_size <= 0:
        raise ValueError("leaf_batch_size must be positive.")

    if max_children is not None and max_children <= 0:
        raise ValueError("max_children must be positive when provided.")

    if search_terminal_value(board) is not None:
        raise ValueError("Cannot run MCTS on a finished game.")

    model.eval()

    stats = MCTSStats()
    start_time = time.perf_counter()
    deadline = None if time_limit_s is None else start_time + time_limit_s

    root = MCTSNode(prior=1.0)
    root_value = evaluate_and_expand(
        node=root,
        board=board,
        model=model,
        cache=eval_cache,
        stats=stats,
        max_children=max_children,
    )

    if root_dirichlet_alpha is not None:
        add_root_dirichlet_noise(
            root=root,
            alpha=root_dirichlet_alpha,
            exploration_fraction=root_exploration_fraction,
        )

    while _should_continue_search(
        simulations_run=stats.simulations_run,
        num_simulations=num_simulations,
        deadline=deadline,
    ):
        if leaf_batch_size <= 1:
            run_single_leaf_simulation(
                root=root,
                board=board,
                model=model,
                c_puct=c_puct,
                eval_cache=eval_cache,
                stats=stats,
                max_children=max_children,
            )
        else:
            run_batched_leaf_simulations(
                root=root,
                board=board,
                model=model,
                c_puct=c_puct,
                eval_cache=eval_cache,
                stats=stats,
                max_children=max_children,
                leaf_batch_size=leaf_batch_size,
                num_simulations=num_simulations,
                deadline=deadline,
            )

    stats.elapsed_seconds = time.perf_counter() - start_time

    visit_counts = {
        move: child.visit_count
        for move, child in root.children.items()
    }

    total_visits = sum(visit_counts.values())

    if total_visits == 0:
        if not root.children:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available from MCTS root.")
            move = legal_moves[0]
        else:
            move = max(
                root.children.items(),
                key=lambda item: item[1].prior,
            )[0]

        policy_target = make_policy_target({move: 1})
    else:
        policy_target = make_policy_target(visit_counts)
        move = select_move_from_visits(
            visit_counts=visit_counts,
            temperature=temperature,
        )

    return MCTSResult(
        move=move,
        visit_counts=visit_counts,
        policy_target=policy_target,
        root_value=root_value,
        simulations_run=stats.simulations_run,
        elapsed_seconds=stats.elapsed_seconds,
        model_evaluations=stats.model_evaluations,
        model_batches=stats.model_batches,
        avg_model_batch_size=stats.avg_model_batch_size,
        max_model_batch_size=stats.max_model_batch_size,
        cache_hits=stats.cache_hits,
        cache_misses=stats.cache_misses,
    )

