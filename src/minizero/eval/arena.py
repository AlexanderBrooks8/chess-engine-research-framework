from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from minizero.engine.base import BaseEngine

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def load_opening_fens(path: str | Path) -> list[str]:
    path = Path(path)
    fens: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            fen = line.strip()

            if not fen or fen.startswith("#"):
                continue

            try:
                chess.Board(fen)
            except ValueError as exc:
                raise ValueError(f"Invalid FEN in {path}: {fen}") from exc

            fens.append(fen)

    if not fens:
        raise ValueError(f"No opening FENs found in {path}.")

    return fens


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


@dataclass
class GameResult:
    white_engine: str
    black_engine: str
    result: str
    winner: str | None
    plies: int
    termination: str
    pgn_game: chess.pgn.Game


@dataclass
class MatchResult:
    white_engine: str
    black_engine: str
    games: int
    white_wins: int
    black_wins: int
    draws: int
    avg_plies: float
    termination_counts: dict[str, int]


def play_game(
    white_engine,
    black_engine,
    max_plies: int = 256,
    adjudicate_material: bool = False,
    adjudication_threshold: float = 1.0,
    start_fen: str | None = None,
) -> GameResult:
    board = chess.Board(start_fen) if start_fen is not None else chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["White"] = white_engine.name
    pgn_game.headers["Black"] = black_engine.name

    if start_fen is not None and start_fen != chess.STARTING_FEN:
        pgn_game.headers["SetUp"] = "1"
        pgn_game.headers["FEN"] = start_fen

    pgn_node = pgn_game
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = engine.choose_move(board)

        if move not in board.legal_moves:
            raise ValueError(f"{engine.name} produced illegal move: {move}")

        board.push(move)
        pgn_node = pgn_node.add_variation(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        if adjudicate_material:
            result = adjudicate_result_by_material(
                board,
                threshold=adjudication_threshold,
            )
            termination = "max_plies_adjudicated_material"
        else:
            result = "1/2-1/2"
            termination = "max_plies"
    else:
        result = board.result(claim_draw=True)
        termination = str(outcome.termination)

    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = None

    pgn_game.headers["Result"] = result
    pgn_game.headers["Termination"] = termination
    pgn_game.headers["PlyCount"] = str(plies)

    return GameResult(
        white_engine=white_engine.name,
        black_engine=black_engine.name,
        result=result,
        winner=winner,
        plies=plies,
        termination=termination,
        pgn_game=pgn_game,
    )


def run_match(
    white_engine,
    black_engine,
    games: int,
    max_plies: int = 256,
    pgn_path: str | Path | None = None,
    adjudicate_material: bool = False,
    adjudication_threshold: float = 1.0,
    opening_fens: list[str] | None = None,
) -> MatchResult:
    results = []

    for game_index in range(games):
        start_fen = None

        if opening_fens is not None:
            if not opening_fens:
                raise ValueError("opening_fens must not be empty when provided.")

            start_fen = opening_fens[game_index % len(opening_fens)]

        results.append(
            play_game(
                white_engine,
                black_engine,
                max_plies=max_plies,
                adjudicate_material=adjudicate_material,
                adjudication_threshold=adjudication_threshold,
                start_fen=start_fen,
            )
        )

    if pgn_path is not None:
        pgn_path = Path(pgn_path)
        pgn_path.parent.mkdir(parents=True, exist_ok=True)

        with pgn_path.open("w", encoding="utf-8") as file:
            for result in results:
                print(result.pgn_game, file=file, end="\n\n")

    white_wins = sum(result.winner == "white" for result in results)
    black_wins = sum(result.winner == "black" for result in results)
    draws = sum(result.winner is None for result in results)
    avg_plies = sum(result.plies for result in results) / games

    termination_counts: dict[str, int] = {}

    for result in results:
        termination_counts[result.termination] = (
            termination_counts.get(result.termination, 0) + 1
        )

    return MatchResult(
        white_engine=white_engine.name,
        black_engine=black_engine.name,
        games=games,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        avg_plies=avg_plies,
        termination_counts=termination_counts,
    )