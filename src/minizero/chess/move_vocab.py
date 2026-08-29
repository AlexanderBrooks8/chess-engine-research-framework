import chess
import torch

PROMOTION_PIECES = ["q", "r", "b", "n"]

def build_uci_vocab() -> list[str]:
    moves: list[str] = []

    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue

            base = chess.square_name(from_square) + chess.square_name(to_square)
            moves.append(base)

            from_rank = chess.square_rank(from_square)
            to_rank = chess.square_rank(to_square)

            is_promotion_rank = (
                (from_rank == 6 and to_rank == 7)
                or (from_rank == 1 and to_rank == 0)
            )

            if is_promotion_rank:
                for promo in PROMOTION_PIECES:
                    moves.append(base + promo)

    return sorted(set(moves))

UCI_MOVES = build_uci_vocab()
MOVE_TO_ID = {move: idx for idx, move in enumerate(UCI_MOVES)}
ID_TO_MOVE = {idx: move for move, idx in MOVE_TO_ID.items()}
VOCAB_SIZE = len(UCI_MOVES)

def move_to_id(move: chess.Move) -> int:
    return MOVE_TO_ID[move.uci()]


def id_to_move(move_id: int) -> chess.Move:
    return chess.Move.from_uci(ID_TO_MOVE[move_id])


def legal_move_ids(board: chess.Board) -> list[int]:
    return [move_to_id(move) for move in board.legal_moves]


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    ids = legal_move_ids(board)
    mask[ids] = True
    return mask