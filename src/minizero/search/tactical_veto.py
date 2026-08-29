from __future__ import annotations

from dataclasses import dataclass

import chess


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

MATE_SCORE = 10_000.0


@dataclass(frozen=True)
class TacticalVetoDecision:
    selected_move: chess.Move
    original_move: chess.Move
    minimax_best_move: chess.Move
    original_score: float
    selected_score: float
    best_score: float
    original_loss: float
    was_vetoed: bool
    candidates_checked: int


def material_score(board: chess.Board, perspective: chess.Color) -> float:
    score = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            continue

        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value

    return score if perspective == chess.WHITE else -score


def static_eval(
    board: chess.Board,
    perspective: chess.Color,
    mobility_weight: float,
) -> float:
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == perspective else MATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = material_score(board, perspective=perspective)

    if mobility_weight != 0.0:
        mobility = float(board.legal_moves.count())
        score += mobility_weight * mobility if board.turn == perspective else -mobility_weight * mobility

    return score


def cache_key(
    board: chess.Board,
    depth: int,
    perspective: chess.Color,
) -> tuple[str, int, bool]:
    # Ignore halfmove/fullmove counters for transposition reuse.
    fen_parts = board.fen().split(" ")
    key_fen = " ".join(fen_parts[:4])
    return key_fen, depth, perspective


def minimax_score(
    board: chess.Board,
    depth: int,
    perspective: chess.Color,
    mobility_weight: float,
    cache: dict[tuple[str, int, bool], float],
    alpha: float = -float("inf"),
    beta: float = float("inf"),
) -> float:
    key = cache_key(board, depth, perspective)

    if key in cache:
        return cache[key]

    if depth <= 0 or board.is_game_over(claim_draw=True):
        value = static_eval(
            board=board,
            perspective=perspective,
            mobility_weight=mobility_weight,
        )
        cache[key] = value
        return value

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        value = static_eval(
            board=board,
            perspective=perspective,
            mobility_weight=mobility_weight,
        )
        cache[key] = value
        return value

    if board.turn == perspective:
        best = -float("inf")

        for move in legal_moves:
            board.push(move)
            value = minimax_score(
                board=board,
                depth=depth - 1,
                perspective=perspective,
                mobility_weight=mobility_weight,
                cache=cache,
                alpha=alpha,
                beta=beta,
            )
            board.pop()

            best = max(best, value)
            alpha = max(alpha, best)

            if alpha >= beta:
                break
    else:
        best = float("inf")

        for move in legal_moves:
            board.push(move)
            value = minimax_score(
                board=board,
                depth=depth - 1,
                perspective=perspective,
                mobility_weight=mobility_weight,
                cache=cache,
                alpha=alpha,
                beta=beta,
            )
            board.pop()

            best = min(best, value)
            beta = min(beta, best)

            if alpha >= beta:
                break

    cache[key] = best
    return best


def score_candidate_move(
    board: chess.Board,
    move: chess.Move,
    depth: int,
    perspective: chess.Color,
    mobility_weight: float,
    cache: dict[tuple[str, int, bool], float],
) -> float:
    if move not in board.legal_moves:
        raise ValueError(f"Cannot score illegal move: {move}")

    board.push(move)
    score = minimax_score(
        board=board,
        depth=max(depth - 1, 0),
        perspective=perspective,
        mobility_weight=mobility_weight,
        cache=cache,
    )
    board.pop()

    return score


def sorted_visit_candidates(
    visit_counts: dict[chess.Move, int],
    preferred_move: chess.Move,
    top_n: int,
) -> list[chess.Move]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    candidates = [
        move
        for move, _ in sorted(
            visit_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
    ]

    if preferred_move not in candidates:
        candidates.insert(0, preferred_move)

    # Preserve order while removing duplicates.
    unique: list[chess.Move] = []
    seen: set[chess.Move] = set()

    for move in candidates:
        if move in seen:
            continue
        unique.append(move)
        seen.add(move)

    return unique


def choose_tactical_veto_move(
    board: chess.Board,
    preferred_move: chess.Move,
    visit_counts: dict[chess.Move, int],
    top_n: int = 8,
    depth: int = 2,
    threshold_pawns: float = 2.0,
    mobility_weight: float = 0.01,
) -> TacticalVetoDecision:
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    if threshold_pawns < 0:
        raise ValueError("threshold_pawns must be non-negative.")

    if preferred_move not in board.legal_moves:
        raise ValueError(f"Preferred move is illegal: {preferred_move}")

    perspective = board.turn
    cache: dict[tuple[str, int, bool], float] = {}
    candidates = sorted_visit_candidates(
        visit_counts=visit_counts,
        preferred_move=preferred_move,
        top_n=top_n,
    )

    scored: list[tuple[chess.Move, float]] = []

    for move in candidates:
        if move not in board.legal_moves:
            continue

        scored.append(
            (
                move,
                score_candidate_move(
                    board=board,
                    move=move,
                    depth=depth,
                    perspective=perspective,
                    mobility_weight=mobility_weight,
                    cache=cache,
                ),
            )
        )

    if not scored:
        raise ValueError("No legal tactical-veto candidates available.")

    score_by_move = dict(scored)
    minimax_best_move, best_score = max(scored, key=lambda item: item[1])
    original_score = score_by_move[preferred_move]
    original_loss = max(0.0, best_score - original_score)

    if original_loss <= threshold_pawns:
        return TacticalVetoDecision(
            selected_move=preferred_move,
            original_move=preferred_move,
            minimax_best_move=minimax_best_move,
            original_score=original_score,
            selected_score=original_score,
            best_score=best_score,
            original_loss=original_loss,
            was_vetoed=False,
            candidates_checked=len(scored),
        )

    # Keep the MCTS ordering as much as possible. Do not blindly choose the
    # minimax-best move unless every higher-visit move fails the safety threshold.
    selected_move = minimax_best_move
    selected_score = best_score

    for move, score in scored:
        loss = max(0.0, best_score - score)

        if loss <= threshold_pawns:
            selected_move = move
            selected_score = score
            break

    return TacticalVetoDecision(
        selected_move=selected_move,
        original_move=preferred_move,
        minimax_best_move=minimax_best_move,
        original_score=original_score,
        selected_score=selected_score,
        best_score=best_score,
        original_loss=original_loss,
        was_vetoed=True,
        candidates_checked=len(scored),
    )
