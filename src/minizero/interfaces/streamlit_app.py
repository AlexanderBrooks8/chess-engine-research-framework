from __future__ import annotations

import argparse
from pathlib import Path

import chess
import streamlit as st

from minizero.engine.random_engine import RandomEngine
from minizero.engine.material_engine import MaterialEngine
from minizero.engine.tactical_engine import TacticalEngine
from minizero.engine.neural_engine import NeuralEngine


ENGINES = {
    "random": RandomEngine,
    "material": MaterialEngine,
    "tactical": TacticalEngine,
    "neural": NeuralEngine,
}


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--neural-device", "--device", default="cuda")
    args, _ = parser.parse_known_args()
    return args


CLI_ARGS = parse_cli_args()


@st.cache_resource(show_spinner=False)
def get_cached_engine(
    engine_name: str,
    checkpoint_path: str,
    neural_device: str,
):
    if engine_name == "neural":
        if not checkpoint_path:
            raise ValueError("Neural engine requires a checkpoint path.")

        return NeuralEngine(
            checkpoint_path=Path(checkpoint_path),
            device=neural_device,
        )

    return ENGINES[engine_name]()


def get_engine(engine_name: str, checkpoint_path: str, neural_device: str):
    return get_cached_engine(engine_name, checkpoint_path, neural_device)


def init_state() -> None:
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()

    if "move_history" not in st.session_state:
        st.session_state.move_history = []

    if "selected_square" not in st.session_state:
        st.session_state.selected_square = None



def reset_game() -> None:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.selected_square = None



def square_label(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)

    if piece is None:
        return "·"

    return piece.unicode_symbol()



def make_candidate_move(
    board: chess.Board,
    from_square: chess.Square,
    to_square: chess.Square,
) -> chess.Move:
    piece = board.piece_at(from_square)

    is_promotion = (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(to_square) in {0, 7}
    )

    if is_promotion:
        return chess.Move(from_square, to_square, promotion=chess.QUEEN)

    return chess.Move(from_square, to_square)



def handle_square_click(square: chess.Square, human_color: str) -> None:
    board: chess.Board = st.session_state.board
    selected_square = st.session_state.selected_square

    if board.is_game_over(claim_draw=True):
        return

    human_is_white = human_color == "white"
    human_turn = board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK

    if not human_turn:
        return

    if selected_square is None:
        piece = board.piece_at(square)

        if piece is None:
            return

        if piece.color != board.turn:
            return

        st.session_state.selected_square = square
        return

    move = make_candidate_move(board, selected_square, square)

    if move not in board.legal_moves:
        st.session_state.selected_square = None
        return

    push_move(move, "Human")
    st.session_state.selected_square = None



def inject_board_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="column"] button {
            height: 72px;
            min-height: 72px;
            font-size: 34px;
            padding: 0px;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_board(board: chess.Board, human_color: str) -> None:
    selected_square = st.session_state.selected_square

    ranks = range(7, -1, -1) if human_color == "white" else range(8)
    files = range(8) if human_color == "white" else range(7, -1, -1)

    for rank in ranks:
        cols = st.columns(8)

        for col_idx, file in enumerate(files):
            square = chess.square(file, rank)
            label = square_label(board, square)

            if square == selected_square:
                label = f"[{label}]"

            square_name = chess.square_name(square)

            with cols[col_idx]:
                if st.button(
                    label,
                    key=f"square_{square_name}",
                    use_container_width=True,
                ):
                    handle_square_click(square, human_color)
                    st.rerun()



def push_move(move: chess.Move, label: str) -> None:
    san = st.session_state.board.san(move)
    st.session_state.board.push(move)
    st.session_state.move_history.append(f"{label}: {san}")



def make_engine_move(engine_name: str, checkpoint_path: str, neural_device: str) -> None:
    board = st.session_state.board

    if board.is_game_over():
        return

    try:
        engine = get_engine(engine_name, checkpoint_path, neural_device)
    except Exception as exc:
        st.error(f"Could not load engine: {exc}")
        return

    move = engine.choose_move(board)

    if move not in board.legal_moves:
        st.error(f"Engine produced illegal move: {move}")
        return

    push_move(move, "Engine")



def make_human_move(move_text: str) -> None:
    board = st.session_state.board

    if board.is_game_over():
        st.warning("Game is already over.")
        return

    try:
        move = chess.Move.from_uci(move_text.strip())
    except ValueError:
        st.error("Invalid move format. Use UCI like e2e4 or g1f3.")
        return

    if move not in board.legal_moves:
        st.error("Illegal move.")
        return

    push_move(move, "Human")



def main() -> None:
    st.set_page_config(page_title="AlexZero Chess", layout="wide")
    init_state()
    inject_board_css()

    st.title("AlexZero Chess GUI")

    with st.sidebar:
        st.header("Settings")

        engine_name = st.selectbox(
            "Engine",
            options=list(ENGINES.keys()),
            index=3,
        )

        default_checkpoint = CLI_ARGS.checkpoint or "checkpoints/diagnostic_10m_cnn192x8_sparsecache_pv10_vw1.pt"
        checkpoint_path = st.text_input(
            "Neural checkpoint",
            value=default_checkpoint,
            disabled=engine_name != "neural",
        )

        neural_device = st.selectbox(
            "Neural device",
            options=["cuda", "cpu"],
            index=0 if CLI_ARGS.neural_device == "cuda" else 1,
            disabled=engine_name != "neural",
        )

        human_color = st.selectbox(
            "Human color",
            options=["white", "black"],
            index=0,
        )

        if st.button("Clear cached engines"):
            get_cached_engine.clear()
            st.success("Cleared cached engines. The next engine move will reload the checkpoint.")

        if st.button("Reset game"):
            reset_game()
            st.rerun()

    board: chess.Board = st.session_state.board
    human_is_white = human_color == "white"
    human_turn = board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK

    left_col, right_col = st.columns([3, 1])

    with left_col:
        render_board(board, human_color)

    with right_col:
        st.subheader("Game state")

        st.write(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        st.write(f"Result: {board.result(claim_draw=True)}")

        if board.is_game_over(claim_draw=True):
            st.success(f"Game over: {board.outcome(claim_draw=True)}")

        st.text_area("FEN", value=board.fen(), height=80)

        st.subheader("Move input")

        if human_turn:
            selected_square = st.session_state.selected_square

            if selected_square is None:
                st.info("Click one of your pieces.")
            else:
                st.info(f"Selected: {chess.square_name(selected_square)}. Click a target square.")

            move_text = st.text_input("Optional UCI input", placeholder="e2e4")

            if st.button("Play typed move"):
                make_human_move(move_text)
                st.session_state.selected_square = None
                st.rerun()
        else:
            st.info("Engine to move.")

            if st.button("Play engine move"):
                make_engine_move(engine_name, checkpoint_path, neural_device)
                st.session_state.selected_square = None
                st.rerun()

        st.subheader("Move history")
        if st.session_state.move_history:
            st.write("\n".join(st.session_state.move_history[-30:]))
        else:
            st.write("No moves yet.")


if __name__ == "__main__":
    main()
