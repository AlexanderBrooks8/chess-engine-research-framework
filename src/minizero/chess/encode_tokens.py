from __future__ import annotations

import chess
import torch


EMPTY = 0

PIECE_TO_TOKEN = {
    chess.Piece(chess.PAWN, chess.WHITE): 1,
    chess.Piece(chess.KNIGHT, chess.WHITE): 2,
    chess.Piece(chess.BISHOP, chess.WHITE): 3,
    chess.Piece(chess.ROOK, chess.WHITE): 4,
    chess.Piece(chess.QUEEN, chess.WHITE): 5,
    chess.Piece(chess.KING, chess.WHITE): 6,
    chess.Piece(chess.PAWN, chess.BLACK): 7,
    chess.Piece(chess.KNIGHT, chess.BLACK): 8,
    chess.Piece(chess.BISHOP, chess.BLACK): 9,
    chess.Piece(chess.ROOK, chess.BLACK): 10,
    chess.Piece(chess.QUEEN, chess.BLACK): 11,
    chess.Piece(chess.KING, chess.BLACK): 12,
}

PIECE_VOCAB_SIZE = 13


def encode_board_tokens(board: chess.Board) -> torch.Tensor:
    tokens = torch.zeros(64, dtype=torch.long)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tokens[square] = EMPTY if piece is None else PIECE_TO_TOKEN[piece]

    return tokens


def encode_side_to_move(board: chess.Board) -> torch.Tensor:
    return torch.tensor(0 if board.turn == chess.WHITE else 1, dtype=torch.long)


def encode_castling_rights(board: chess.Board) -> torch.Tensor:
    rights = 0

    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 8

    return torch.tensor(rights, dtype=torch.long)


def encode_en_passant(board: chess.Board) -> torch.Tensor:
    square = board.ep_square
    return torch.tensor(64 if square is None else square, dtype=torch.long)


def encode_position(board: chess.Board) -> dict[str, torch.Tensor]:
    return {
        "board_tokens": encode_board_tokens(board),
        "side_to_move": encode_side_to_move(board),
        "castling_rights": encode_castling_rights(board),
        "en_passant": encode_en_passant(board),
    }