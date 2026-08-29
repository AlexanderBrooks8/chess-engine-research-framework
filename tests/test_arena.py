import chess

from minizero.engine.random_engine import RandomEngine
from minizero.engine.material_engine import MaterialEngine
from minizero.eval.arena import (
    adjudicate_result_by_material,
    load_opening_fens,
    material_score,
    play_game,
    run_match,
)


def test_play_game_returns_result():
    result = play_game(RandomEngine(), RandomEngine(), max_plies=20)

    assert result.white_engine == "random"
    assert result.black_engine == "random"
    assert result.result in {"1-0", "0-1", "1/2-1/2", "*"}
    assert result.plies <= 20


def test_run_match_counts_games():
    result = run_match(RandomEngine(), MaterialEngine(), games=4, max_plies=20)

    assert result.games == 4
    assert result.white_wins + result.black_wins + result.draws == 4
    assert result.avg_plies <= 20
    
def test_arena_material_score_starting_position_is_equal():
    board = chess.Board()

    assert material_score(board) == 0.0


def test_arena_adjudicate_result_by_material_white_ahead():
    board = chess.Board()
    board.remove_piece_at(chess.D8)

    assert adjudicate_result_by_material(board) == "1-0"


def test_arena_adjudicate_result_by_material_black_ahead():
    board = chess.Board()
    board.remove_piece_at(chess.D1)

    assert adjudicate_result_by_material(board) == "0-1"


def test_arena_adjudicate_result_by_material_equal_is_draw():
    board = chess.Board()

    assert adjudicate_result_by_material(board) == "1/2-1/2"

def test_play_game_accepts_start_fen():
    start_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    result = play_game(
        RandomEngine(),
        RandomEngine(),
        max_plies=2,
        start_fen=start_fen,
    )

    assert result.pgn_game.headers["SetUp"] == "1"
    assert result.pgn_game.headers["FEN"] == start_fen
    assert result.plies <= 2


def test_run_match_cycles_opening_fens():
    opening_fens = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    ]

    result = run_match(
        RandomEngine(),
        RandomEngine(),
        games=3,
        max_plies=2,
        opening_fens=opening_fens,
    )

    assert result.games == 3
    assert result.white_wins + result.black_wins + result.draws == 3


def test_load_opening_fens_ignores_comments_and_blanks(tmp_path):
    path = tmp_path / "openings.txt"
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    path.write_text(f"# comment\n\n{fen}\n", encoding="utf-8")

    assert load_opening_fens(path) == [fen]
