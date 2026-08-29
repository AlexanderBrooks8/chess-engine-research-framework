from __future__ import annotations

import chess

from minizero.engine.random_engine import RandomEngine
from minizero.engine.material_engine import MaterialEngine
from minizero.engine.tactical_engine import TacticalEngine
from minizero.engine.neural_engine import NeuralEngine


ENGINES = {
    "random": RandomEngine,
    "material": MaterialEngine,
    "tactical": TacticalEngine,
    "neural": NeuralEngine,
    "neural_sample": lambda: NeuralEngine(deterministic=False, temperature=1.0),
}


def print_board(board: chess.Board) -> None:
    print()
    print(board)
    print()
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print()


def get_human_move(board: chess.Board) -> chess.Move:
    while True:
        move_text = input("Your move, UCI format e.g. e2e4: ").strip()

        if move_text.lower() in {"quit", "exit"}:
            raise KeyboardInterrupt

        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            print("Invalid UCI format. Try something like e2e4 or g1f3.")
            continue

        if move not in board.legal_moves:
            print("Illegal move. Try again.")
            continue

        return move


def play_cli(engine_name: str = "material", human_color: str = "white") -> None:
    if engine_name not in ENGINES:
        raise ValueError(f"Unknown engine '{engine_name}'. Options: {list(ENGINES)}")

    board = chess.Board()
    engine = ENGINES[engine_name]()

    human_is_white = human_color.lower() == "white"

    print(f"Starting game vs {engine.name} agent.")
    print("Type 'quit' or 'exit' to stop.")

    try:
        while not board.is_game_over():
            print_board(board)

            human_turn = board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK

            if human_turn:
                move = get_human_move(board)
            else:
                move = engine.choose_move(board)
                print(f"Engine move: {move.uci()}")

            board.push(move)

        print_board(board)
        print(f"Game over: {board.result()}")
        print(f"Reason: {board.outcome()}")

    except KeyboardInterrupt:
        print("\nGame stopped.")


if __name__ == "__main__":
    play_cli()