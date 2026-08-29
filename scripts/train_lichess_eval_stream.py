from __future__ import annotations

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Iterable

import chess
import torch
import torch.nn.functional as F

from minizero.chess.encode_tokens import encode_position
from minizero.chess.move_vocab import VOCAB_SIZE, legal_move_mask, move_to_id
from minizero.models.cnn_policy_value import CNNPolicyValue
from minizero.models.factory import load_transformer_from_checkpoint
from minizero.models.model_specs import CNN_CONFIG, MODEL_CONFIG, MODEL_TYPE
from minizero.models.transformer_policy_value import TransformerPolicyValue


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

MAX_MATERIAL_BALANCE = 39.0
MAX_LEGAL_MOBILITY = 80.0
MAX_KING_ZONE_SQUARES = 9.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_cp(cp: int, cp_scale: float) -> float:
    return math.tanh(float(cp) / cp_scale)


def normalize_mate(mate: int) -> float:
    if mate == 0:
        return 0.0

    return 1.0 if mate > 0 else -1.0


def score_to_value(pv: dict[str, Any], cp_scale: float) -> float:
    if "cp" in pv:
        return normalize_cp(int(pv["cp"]), cp_scale=cp_scale)

    if "mate" in pv:
        return normalize_mate(int(pv["mate"]))

    raise ValueError(f"PV has neither cp nor mate score: {pv}")


def piece_value(piece_type: chess.PieceType | None) -> float:
    if piece_type is None:
        return 0.0
    return PIECE_VALUES[piece_type]


def captured_piece_type(board: chess.Board, move: chess.Move) -> chess.PieceType | None:
    if board.is_en_passant(move):
        return chess.PAWN

    captured = board.piece_at(move.to_square)
    if captured is None:
        return None

    return captured.piece_type


def material_balance_for_side_to_move(board: chess.Board) -> float:
    """Return side-to-move material balance normalized to roughly [-1, 1]."""
    white_material = 0.0
    black_material = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            continue

        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value

    white_balance = white_material - black_material
    side_to_move_balance = white_balance if board.turn == chess.WHITE else -white_balance

    return max(-1.0, min(1.0, side_to_move_balance / MAX_MATERIAL_BALANCE))


def side_to_move_has_mate_in_1(board: chess.Board) -> float:
    """Return 1.0 when the side to move has an immediate checkmate available."""
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)

        if child.is_checkmate():
            return 1.0

    return 0.0


def side_to_move_aux_targets(board: chess.Board) -> dict[str, float]:
    """Compute cheap, board-derived auxiliary targets without minimax/search.

    All regression targets except material are normalized to [0, 1]. Binary targets
    are floats so they can be consumed by BCEWithLogitsLoss.
    """
    side = board.turn
    opponent = not side
    legal_moves = list(board.legal_moves)

    legal_mobility = clamp01(len(legal_moves) / MAX_LEGAL_MOBILITY)
    in_check = 1.0 if board.is_check() else 0.0
    has_check = 1.0 if any(board.gives_check(move) for move in legal_moves) else 0.0
    capture_available = 1.0 if any(board.is_capture(move) for move in legal_moves) else 0.0

    best_capture_value = 0.0
    for move in legal_moves:
        if not board.is_capture(move):
            continue
        best_capture_value = max(
            best_capture_value,
            piece_value(captured_piece_type(board, move)),
        )

    attack_pressure = 0.0
    hanging_material = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None or piece.color != side:
            continue

        value = PIECE_VALUES[piece.piece_type]
        attacked_by_opponent = bool(board.attackers(opponent, square))

        if not attacked_by_opponent:
            continue

        attack_pressure += value

        defended_by_side = bool(board.attackers(side, square))
        if not defended_by_side:
            hanging_material += value

    king_pressure = 0.0
    king_square = board.king(side)
    if king_square is not None:
        king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square])
        attacked_zone_squares = sum(
            1 for square in king_zone if board.is_attacked_by(opponent, square)
        )
        king_pressure = clamp01(attacked_zone_squares / MAX_KING_ZONE_SQUARES)

    return {
        "target_material": material_balance_for_side_to_move(board),
        "target_mate_in_1": side_to_move_has_mate_in_1(board),
        "target_in_check": in_check,
        "target_has_check": has_check,
        "target_capture_available": capture_available,
        "target_legal_mobility": legal_mobility,
        "target_attack_pressure": clamp01(attack_pressure / MAX_MATERIAL_BALANCE),
        "target_king_pressure": king_pressure,
        "target_best_capture": clamp01(best_capture_value / PIECE_VALUES[chess.QUEEN]),
        "target_hanging_material": clamp01(hanging_material / MAX_MATERIAL_BALANCE),
    }

AUX_TARGET_NAMES = (
    "target_material",
    "target_mate_in_1",
    "target_in_check",
    "target_has_check",
    "target_capture_available",
    "target_legal_mobility",
    "target_attack_pressure",
    "target_king_pressure",
    "target_best_capture",
    "target_hanging_material",
)


def zero_aux_targets() -> dict[str, float]:
    return {name: 0.0 for name in AUX_TARGET_NAMES}


def aux_heads_enabled(
    material_weight: float,
    mate_in_1_weight: float,
    in_check_weight: float,
    has_check_weight: float,
    capture_available_weight: float,
    legal_mobility_weight: float,
    attack_pressure_weight: float,
    king_pressure_weight: float,
    best_capture_weight: float,
    hanging_material_weight: float,
) -> bool:
    return any(
        weight > 0.0
        for weight in (
            material_weight,
            mate_in_1_weight,
            in_check_weight,
            has_check_weight,
            capture_available_weight,
            legal_mobility_weight,
            attack_pressure_weight,
            king_pressure_weight,
            best_capture_weight,
            hanging_material_weight,
        )
    )


_FAST_FEN_TABLES: dict[str, Any] | None = None


def _metadata_scalar(value: torch.Tensor) -> int:
    return int(value.detach().cpu().view(-1)[0].item())


def _infer_fast_fen_tables() -> dict[str, Any]:
    """Infer token/meta mappings from encode_position once, outside the hot path.

    This keeps the fast FEN parser aligned with minizero.chess.encode_tokens even if
    token IDs or metadata IDs change later. The training hot path then avoids
    constructing a python-chess Board for every streamed position.
    """
    piece_to_token: dict[str, int] = {}

    empty_board = chess.Board(None)
    empty_encoded = encode_position(empty_board)
    empty_token = int(empty_encoded["board_tokens"][0].item())

    for symbol in "PNBRQKpnbrqk":
        board = chess.Board(None)
        board.set_piece_at(chess.A1, chess.Piece.from_symbol(symbol))
        encoded = encode_position(board)
        piece_to_token[symbol] = int(encoded["board_tokens"][chess.A1].item())

    side_to_move = {
        "w": _metadata_scalar(encode_position(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))["side_to_move"]),
        "b": _metadata_scalar(encode_position(chess.Board("8/8/8/8/8/8/8/8 b - - 0 1"))["side_to_move"]),
    }

    castling_rights: dict[str, int] = {}
    rights_symbols = "KQkq"
    for mask in range(16):
        rights = "".join(symbol for bit, symbol in enumerate(rights_symbols) if mask & (1 << bit))
        fen_rights = rights if rights else "-"
        board = chess.Board(f"8/8/8/8/8/8/8/8 w {fen_rights} - 0 1")
        normalized_key = "".join(symbol for symbol in rights_symbols if symbol in rights) or "-"
        castling_rights[normalized_key] = _metadata_scalar(encode_position(board)["castling_rights"])

    en_passant: dict[str, int] = {}
    for ep in ["-"] + list(chess.SQUARE_NAMES):
        board = chess.Board(f"8/8/8/8/8/8/8/8 w - {ep} 0 1")
        en_passant[ep] = _metadata_scalar(encode_position(board)["en_passant"])

    return {
        "empty_token": empty_token,
        "piece_to_token": piece_to_token,
        "side_to_move": side_to_move,
        "castling_rights": castling_rights,
        "en_passant": en_passant,
    }


def fast_fen_tables() -> dict[str, Any]:
    global _FAST_FEN_TABLES
    if _FAST_FEN_TABLES is None:
        _FAST_FEN_TABLES = _infer_fast_fen_tables()
    return _FAST_FEN_TABLES


def normalize_castling_fen(rights: str) -> str:
    if rights == "-":
        return "-"
    normalized = "".join(symbol for symbol in "KQkq" if symbol in rights)
    return normalized if normalized else "-"


def castling_bitmask_from_fen(castling: str) -> int:
    if castling != "-" and any(char not in "KQkq" for char in castling):
        raise ValueError(f"Invalid FEN castling rights: {castling!r}")

    value = 0
    if "K" in castling:
        value |= 1
    if "Q" in castling:
        value |= 2
    if "k" in castling:
        value |= 4
    if "q" in castling:
        value |= 8
    return value


def encode_position_from_fen_fast_raw(fen: str) -> tuple[list[int], int, int, int]:
    """Encode FEN to Python-native values for the batch-native fast stream path.

    This avoids per-example torch.tensor/torch.full calls. stack_batch() converts
    an entire batch of these Python values to device tensors in a small number of
    bulk tensor creations.
    """
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"Invalid FEN: {fen!r}")

    placement, side, castling, ep_square = fields[:4]
    tables = fast_fen_tables()
    board_tokens = [int(tables["empty_token"])] * 64
    piece_to_token: dict[str, int] = tables["piece_to_token"]

    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN placement: {placement!r}")

    for fen_rank_index, rank_text in enumerate(ranks):
        rank = 7 - fen_rank_index
        file_index = 0
        for char in rank_text:
            if char.isdigit():
                file_index += int(char)
                continue
            if char not in piece_to_token:
                raise ValueError(f"Invalid FEN piece symbol: {char!r}")
            if file_index >= 8:
                raise ValueError(f"Invalid FEN rank overrun: {rank_text!r}")
            square = rank * 8 + file_index
            board_tokens[square] = int(piece_to_token[char])
            file_index += 1
        if file_index != 8:
            raise ValueError(f"Invalid FEN rank width: {rank_text!r}")

    side_map: dict[str, int] = tables["side_to_move"]
    ep_map: dict[str, int] = tables["en_passant"]

    if side not in side_map:
        raise ValueError(f"Invalid FEN side-to-move: {side!r}")
    if ep_square not in ep_map:
        raise ValueError(f"Invalid FEN en-passant square: {ep_square!r}")

    return (
        board_tokens,
        int(side_map[side]),
        castling_bitmask_from_fen(castling),
        int(ep_map[ep_square]),
    )


def encode_position_from_fen_fast(fen: str) -> dict[str, torch.Tensor]:
    """Encode FEN directly, avoiding per-position python-chess Board creation."""
    board_tokens, side_to_move, castling_rights, en_passant = encode_position_from_fen_fast_raw(fen)
    return {
        "board_tokens": torch.tensor(board_tokens, dtype=torch.long),
        "side_to_move": torch.tensor(side_to_move, dtype=torch.long),
        "castling_rights": torch.tensor(castling_rights, dtype=torch.long),
        "en_passant": torch.tensor(en_passant, dtype=torch.long),
    }


def make_trusted_sparse_policy_target_raw(
    pvs: list[dict[str, Any]],
    multipv: int,
    cp_scale: float,
    policy_temperature: float,
) -> tuple[list[int], list[float], float] | None:
    """Sparse PV policy target as Python-native lists, no per-example tensors."""
    if policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive.")

    scored_moves: list[tuple[int, float]] = []
    for pv in pvs[:multipv]:
        line = pv.get("line", "")
        if not line:
            continue

        move_uci = line.split()[0]
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            continue

        value = score_to_value(pv, cp_scale=cp_scale)
        scored_moves.append((move_to_id(move), value))

    if not scored_moves:
        return None

    scaled_values = [value / policy_temperature for _move_id, value in scored_moves]
    max_scaled = max(scaled_values)
    weights = [math.exp(value - max_scaled) for value in scaled_values]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return None

    probs = [weight / total_weight for weight in weights]
    move_ids = [move_id for move_id, _value in scored_moves]
    best_index = max(range(len(probs)), key=probs.__getitem__)
    best_value = scored_moves[best_index][1]
    return move_ids, probs, best_value


def make_trusted_sparse_policy_target(
    pvs: list[dict[str, Any]],
    multipv: int,
    cp_scale: float,
    policy_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Sparse policy target for trusted Stockfish PVs without board legality checks."""
    made = make_trusted_sparse_policy_target_raw(
        pvs=pvs,
        multipv=multipv,
        cp_scale=cp_scale,
        policy_temperature=policy_temperature,
    )
    if made is None:
        return None
    move_ids, probs, best_value = made
    return (
        torch.tensor(move_ids, dtype=torch.long),
        torch.tensor(probs, dtype=torch.float32),
        best_value,
    )


def choose_eval(evals: list[dict[str, Any]], min_depth: int) -> dict[str, Any] | None:
    valid = [
        item
        for item in evals
        if int(item.get("depth", 0)) >= min_depth and item.get("pvs")
    ]

    if not valid:
        return None

    return max(valid, key=lambda item: int(item.get("depth", 0)))


def make_policy_target(
    board: chess.Board,
    pvs: list[dict[str, Any]],
    multipv: int,
    cp_scale: float,
    policy_temperature: float,
) -> tuple[torch.Tensor, float] | None:
    if policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive.")

    scored_moves: list[tuple[chess.Move, float]] = []

    for pv in pvs[:multipv]:
        line = pv.get("line", "")

        if not line:
            continue

        move_uci = line.split()[0]

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            continue

        if move not in board.legal_moves:
            continue

        value = score_to_value(pv, cp_scale=cp_scale)
        scored_moves.append((move, value))

    if not scored_moves:
        return None

    values = torch.tensor(
        [value / policy_temperature for _move, value in scored_moves],
        dtype=torch.float32,
    )
    probs = torch.softmax(values, dim=0)

    policy = torch.zeros(VOCAB_SIZE, dtype=torch.float32)

    for (move, _value), prob in zip(scored_moves, probs):
        policy[move_to_id(move)] = float(prob.item())

    best_index = int(torch.argmax(probs).item())
    best_value = scored_moves[best_index][1]

    return policy, best_value


def make_sparse_policy_target(
    board: chess.Board,
    pvs: list[dict[str, Any]],
    multipv: int,
    cp_scale: float,
    policy_temperature: float,
    trust_stockfish_legal_moves: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Return compact Stockfish PV policy target and best value.

    Sparse mode avoids constructing a dense VOCAB_SIZE policy vector and
    avoids legal_move_mask(board). The loss is computed against the full move
    vocabulary, so this mode is faster but not numerically identical to dense
    legal-masked policy CE.
    """
    if policy_temperature <= 0:
        raise ValueError("policy_temperature must be positive.")

    legal_moves = None if trust_stockfish_legal_moves else set(board.legal_moves)
    scored_moves: list[tuple[int, float]] = []

    for pv in pvs[:multipv]:
        line = pv.get("line", "")

        if not line:
            continue

        move_uci = line.split()[0]

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            continue

        if legal_moves is not None and move not in legal_moves:
            continue

        value = score_to_value(pv, cp_scale=cp_scale)
        scored_moves.append((move_to_id(move), value))

    if not scored_moves:
        return None

    values = torch.tensor(
        [value / policy_temperature for _move_id, value in scored_moves],
        dtype=torch.float32,
    )
    probs = torch.softmax(values, dim=0)
    move_ids = torch.tensor([move_id for move_id, _value in scored_moves], dtype=torch.long)

    best_index = int(torch.argmax(probs).item())
    best_value = scored_moves[best_index][1]

    return move_ids, probs.to(dtype=torch.float32), best_value


def pad_sparse_policy_targets(
    target_move_ids: list[torch.Tensor],
    target_move_probs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not target_move_ids:
        raise ValueError("target_move_ids must not be empty.")

    max_len = max(int(item.numel()) for item in target_move_ids)
    ids = torch.full((len(target_move_ids), max_len), -1, dtype=torch.long)
    probs = torch.zeros((len(target_move_probs), max_len), dtype=torch.float32)

    for row, (row_ids, row_probs) in enumerate(zip(target_move_ids, target_move_probs)):
        n = int(row_ids.numel())
        ids[row, :n] = row_ids.to(dtype=torch.long)
        probs[row, :n] = row_probs.to(dtype=torch.float32)

    return ids, probs


def sparse_policy_ce_loss(
    policy_logits: torch.Tensor,
    target_move_ids: torch.Tensor,
    target_move_probs: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(policy_logits, dim=-1)
    valid = target_move_ids >= 0
    safe_ids = target_move_ids.clamp_min(0)
    selected = log_probs.gather(1, safe_ids)
    return -(selected * target_move_probs * valid.float()).sum(dim=1).mean()


def iter_lines(path: Path) -> Iterable[str]:
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading .zst files requires zstandard. Install with: pip install zstandard"
            ) from exc

        with path.open("rb") as raw_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw_file) as reader:
                buffer = b""

                while True:
                    chunk = reader.read(1024 * 1024)

                    if not chunk:
                        break

                    buffer += chunk

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        yield line.decode("utf-8", errors="replace")

                if buffer:
                    yield buffer.decode("utf-8", errors="replace")

        return

    with path.open("r", encoding="utf-8") as file:
        yield from file


def iter_training_examples(
    input_path: Path,
    max_positions: int,
    min_depth: int,
    multipv: int,
    cp_scale: float,
    policy_temperature: float,
    compute_aux_targets: bool = True,
    sparse_policy_target: bool = False,
    trust_stockfish_legal_moves: bool = False,
    skip_positions: int = 0,
) -> Iterable[dict[str, Any]]:
    if skip_positions < 0:
        raise ValueError("skip_positions must be non-negative.")

    saved = 0
    accepted = 0
    scanned = 0
    use_fast_fen_path = sparse_policy_target and trust_stockfish_legal_moves and not compute_aux_targets

    for raw_line_number, line in enumerate(iter_lines(input_path), start=1):
        if saved >= max_positions:
            break

        scanned = raw_line_number

        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        fen = item.get("fen")
        evals = item.get("evals")

        if not fen or not evals:
            continue

        selected_eval = choose_eval(evals=evals, min_depth=min_depth)

        if selected_eval is None:
            continue

        if use_fast_fen_path:
            try:
                board_tokens, side_to_move, castling_rights, en_passant = encode_position_from_fen_fast_raw(fen)
            except ValueError:
                continue

            made_sparse_raw = make_trusted_sparse_policy_target_raw(
                pvs=selected_eval["pvs"],
                multipv=multipv,
                cp_scale=cp_scale,
                policy_temperature=policy_temperature,
            )

            if made_sparse_raw is None:
                continue

            accepted += 1
            if accepted <= skip_positions:
                if (accepted % 100000) == 0:
                    print("Skipped:", accepted)
                continue

            target_move_ids, target_move_probs, value_target = made_sparse_raw
            saved += 1

            yield {
                "__raw_fast_path": True,
                "board_tokens": board_tokens,
                "side_to_move": side_to_move,
                "castling_rights": castling_rights,
                "en_passant": en_passant,
                "target_move_ids": target_move_ids,
                "target_move_probs": target_move_probs,
                "target_value": float(value_target),
                "saved": saved,
                "scanned": scanned,
            }
            continue

        try:
            board = chess.Board(fen)
        except ValueError:
            continue

        if sparse_policy_target:
            made_sparse = make_sparse_policy_target(
                board=board,
                pvs=selected_eval["pvs"],
                multipv=multipv,
                cp_scale=cp_scale,
                policy_temperature=policy_temperature,
                trust_stockfish_legal_moves=trust_stockfish_legal_moves,
            )

            if made_sparse is None:
                continue

            target_move_ids, target_move_probs, value_target = made_sparse
            policy_target = torch.empty((0,), dtype=torch.float32)
            legal_mask = torch.empty((0,), dtype=torch.bool)
        else:
            made = make_policy_target(
                board=board,
                pvs=selected_eval["pvs"],
                multipv=multipv,
                cp_scale=cp_scale,
                policy_temperature=policy_temperature,
            )

            if made is None:
                continue

            policy_target, value_target = made
            legal_mask = legal_move_mask(board)
            target_move_ids = torch.empty((0,), dtype=torch.long)
            target_move_probs = torch.empty((0,), dtype=torch.float32)

        accepted += 1
        if accepted <= skip_positions:
            continue

        encoded = encode_position(board)
        aux_targets = side_to_move_aux_targets(board) if compute_aux_targets else zero_aux_targets()
        saved += 1

        example: dict[str, Any] = {
            "board_tokens": encoded["board_tokens"],
            "side_to_move": encoded["side_to_move"],
            "castling_rights": encoded["castling_rights"],
            "en_passant": encoded["en_passant"],
            "legal_mask": legal_mask,
            "target_policy": policy_target,
            "target_move_ids": target_move_ids,
            "target_move_probs": target_move_probs,
            "target_value": torch.tensor(value_target, dtype=torch.float32),
            "saved": torch.tensor(saved, dtype=torch.long),
            "scanned": torch.tensor(scanned, dtype=torch.long),
        }

        for name, value in aux_targets.items():
            example[name] = torch.tensor(value, dtype=torch.float32)

        yield example


def make_model(
    checkpoint_in: Path | None,
    model_type: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    ff_dim: int,
    dropout: float,
    cnn_channels: int,
    cnn_blocks: int,
    cnn_dropout: float,
) -> TransformerPolicyValue | CNNPolicyValue:
    if checkpoint_in is not None:
        return load_transformer_from_checkpoint(checkpoint_in)

    normalized_model_type = model_type.lower()

    if normalized_model_type == "transformer":
        return TransformerPolicyValue(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    if normalized_model_type == "cnn":
        return CNNPolicyValue(
            channels=cnn_channels,
            n_blocks=cnn_blocks,
            dropout=cnn_dropout,
        )

    raise ValueError(f"Unknown model_type: {model_type!r}")


def _stack_raw_fast_batch(
    examples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty.")

    batch_size = len(examples)
    max_targets = max(len(item["target_move_ids"]) for item in examples)
    if max_targets <= 0:
        raise ValueError("raw fast examples must contain at least one policy target.")

    board_tokens_np = np.empty((batch_size, 64), dtype=np.int64)
    side_to_move_np = np.empty((batch_size,), dtype=np.int64)
    castling_rights_np = np.empty((batch_size,), dtype=np.int64)
    en_passant_np = np.empty((batch_size,), dtype=np.int64)
    target_value_np = np.empty((batch_size,), dtype=np.float32)
    target_move_ids_np = np.full((batch_size, max_targets), -1, dtype=np.int64)
    target_move_probs_np = np.zeros((batch_size, max_targets), dtype=np.float32)

    for row_idx, item in enumerate(examples):
        board_tokens_np[row_idx, :] = item["board_tokens"]
        side_to_move_np[row_idx] = item["side_to_move"]
        castling_rights_np[row_idx] = item["castling_rights"]
        en_passant_np[row_idx] = item["en_passant"]
        target_value_np[row_idx] = item["target_value"]

        move_ids = item["target_move_ids"]
        move_probs = item["target_move_probs"]
        width = len(move_ids)
        target_move_ids_np[row_idx, :width] = move_ids
        target_move_probs_np[row_idx, :width] = move_probs

    zeros = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    return {
        "board_tokens": torch.from_numpy(board_tokens_np).to(device=device, non_blocking=True),
        "side_to_move": torch.from_numpy(side_to_move_np).to(device=device, non_blocking=True),
        "castling_rights": torch.from_numpy(castling_rights_np).to(device=device, non_blocking=True),
        "en_passant": torch.from_numpy(en_passant_np).to(device=device, non_blocking=True),
        "target_value": torch.from_numpy(target_value_np).to(device=device, non_blocking=True),
        "target_move_ids": torch.from_numpy(target_move_ids_np).to(device=device, non_blocking=True),
        "target_move_probs": torch.from_numpy(target_move_probs_np).to(device=device, non_blocking=True),
        "target_material": zeros,
        "target_mate_in_1": zeros,
        "target_in_check": zeros,
        "target_has_check": zeros,
        "target_capture_available": zeros,
        "target_legal_mobility": zeros,
        "target_attack_pressure": zeros,
        "target_king_pressure": zeros,
        "target_best_capture": zeros,
        "target_hanging_material": zeros,
    }

def stack_batch(
    examples: list[dict[str, Any]],
    device: torch.device,
    sparse_policy_target: bool = False,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty.")

    if examples[0].get("__raw_fast_path") is True:
        return _stack_raw_fast_batch(examples=examples, device=device)

    keys = [
        "board_tokens",
        "side_to_move",
        "castling_rights",
        "en_passant",
        "target_value",
        "target_material",
        "target_mate_in_1",
        "target_in_check",
        "target_has_check",
        "target_capture_available",
        "target_legal_mobility",
        "target_attack_pressure",
        "target_king_pressure",
        "target_best_capture",
        "target_hanging_material",
    ]
    batch = {key: torch.stack([item[key] for item in examples]).to(device) for key in keys}

    if sparse_policy_target:
        target_move_ids, target_move_probs = pad_sparse_policy_targets(
            [item["target_move_ids"] for item in examples],
            [item["target_move_probs"] for item in examples],
        )
        batch["target_move_ids"] = target_move_ids.to(device)
        batch["target_move_probs"] = target_move_probs.to(device)
    else:
        batch["legal_mask"] = torch.stack([item["legal_mask"] for item in examples]).to(device)
        batch["target_policy"] = torch.stack([item["target_policy"] for item in examples]).to(device)

    return batch


def binary_aux_loss(
    output_tensor: torch.Tensor | None,
    target: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if output_tensor is None:
        raise AttributeError(f"Model output has no {name} head.")

    return F.binary_cross_entropy_with_logits(output_tensor.view(-1), target.view(-1))


def regression_aux_loss(
    output_tensor: torch.Tensor | None,
    target: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if output_tensor is None:
        raise AttributeError(f"Model output has no {name} head.")

    return F.mse_loss(output_tensor.view(-1), target.view(-1))


def amp_enabled_for_batch(amp: bool, batch: dict[str, torch.Tensor]) -> bool:
    return bool(amp and batch["board_tokens"].is_cuda)


def scaler_enabled_for_batch(amp: bool, amp_no_scaler: bool, batch: dict[str, torch.Tensor]) -> bool:
    return bool(amp_enabled_for_batch(amp=amp, batch=batch) and not amp_no_scaler)


def make_amp_scaler(device: torch.device, amp: bool, amp_no_scaler: bool = False) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=bool(amp and not amp_no_scaler and device.type == "cuda"))


def configure_tf32(tf32: bool, device: torch.device) -> bool:
    enabled = bool(tf32 and device.type == "cuda")
    if enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return enabled

def train_batch(
    model: TransformerPolicyValue | CNNPolicyValue,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    value_weight: float,
    material_weight: float,
    mate_in_1_weight: float,
    in_check_weight: float,
    has_check_weight: float,
    capture_available_weight: float,
    legal_mobility_weight: float,
    attack_pressure_weight: float,
    king_pressure_weight: float,
    best_capture_weight: float,
    hanging_material_weight: float,
    sparse_policy_target: bool = False,
    return_metrics: bool = True,
    amp: bool = False,
    amp_no_scaler: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    use_amp = amp_enabled_for_batch(amp=amp, batch=batch)
    use_scaler = bool(use_amp and not amp_no_scaler)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        output = model(
            board_tokens=batch["board_tokens"],
            side_to_move=batch["side_to_move"],
            castling_rights=batch["castling_rights"],
            en_passant=batch["en_passant"],
        )

        if sparse_policy_target:
            policy_loss = sparse_policy_ce_loss(
                policy_logits=output.policy_logits,
                target_move_ids=batch["target_move_ids"],
                target_move_probs=batch["target_move_probs"],
            )
        else:
            masked_logits = output.policy_logits.masked_fill(~batch["legal_mask"], float("-inf"))
            log_probs = F.log_softmax(masked_logits, dim=-1)
            log_probs = log_probs.masked_fill(~batch["legal_mask"], 0.0)
            policy_loss = -(batch["target_policy"] * log_probs).sum(dim=-1).mean()

        value_pred = output.value.view(-1)
        value_loss = F.mse_loss(value_pred.float(), batch["target_value"].view(-1).float())

        zero = torch.zeros((), device=batch["target_value"].device)
        losses: dict[str, torch.Tensor] = {
        "policy": policy_loss,
        "value": value_loss,
        "material": zero,
        "mate_in_1": zero,
        "in_check": zero,
        "has_check": zero,
        "capture_available": zero,
        "legal_mobility": zero,
        "attack_pressure": zero,
        "king_pressure": zero,
        "best_capture": zero,
        "hanging_material": zero,
    }

    if material_weight > 0.0:
        losses["material"] = regression_aux_loss(
            output.material,
            batch["target_material"],
            "material",
        )

    if mate_in_1_weight > 0.0:
        losses["mate_in_1"] = binary_aux_loss(
            output.mate_in_1_logits,
            batch["target_mate_in_1"],
            "mate_in_1",
        )

    if in_check_weight > 0.0:
        losses["in_check"] = binary_aux_loss(
            output.in_check_logits,
            batch["target_in_check"],
            "in_check",
        )

    if has_check_weight > 0.0:
        losses["has_check"] = binary_aux_loss(
            output.has_check_logits,
            batch["target_has_check"],
            "has_check",
        )

    if capture_available_weight > 0.0:
        losses["capture_available"] = binary_aux_loss(
            output.capture_available_logits,
            batch["target_capture_available"],
            "capture_available",
        )

    if legal_mobility_weight > 0.0:
        losses["legal_mobility"] = regression_aux_loss(
            output.legal_mobility,
            batch["target_legal_mobility"],
            "legal_mobility",
        )

    if attack_pressure_weight > 0.0:
        losses["attack_pressure"] = regression_aux_loss(
            output.attack_pressure,
            batch["target_attack_pressure"],
            "attack_pressure",
        )

    if king_pressure_weight > 0.0:
        losses["king_pressure"] = regression_aux_loss(
            output.king_pressure,
            batch["target_king_pressure"],
            "king_pressure",
        )

    if best_capture_weight > 0.0:
        losses["best_capture"] = regression_aux_loss(
            output.best_capture,
            batch["target_best_capture"],
            "best_capture",
        )

    if hanging_material_weight > 0.0:
        losses["hanging_material"] = regression_aux_loss(
            output.hanging_material,
            batch["target_hanging_material"],
            "hanging_material",
        )

    loss = (
        losses["policy"]
        + value_weight * losses["value"]
        + material_weight * losses["material"]
        + mate_in_1_weight * losses["mate_in_1"]
        + in_check_weight * losses["in_check"]
        + has_check_weight * losses["has_check"]
        + capture_available_weight * losses["capture_available"]
        + legal_mobility_weight * losses["legal_mobility"]
        + attack_pressure_weight * losses["attack_pressure"]
        + king_pressure_weight * losses["king_pressure"]
        + best_capture_weight * losses["best_capture"]
        + hanging_material_weight * losses["hanging_material"]
    )

    if use_scaler:
        if scaler is None:
            raise ValueError("scaler is required when AMP scaling is enabled.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    if not return_metrics:
        return {}

    result = {name: float(value.detach().item()) for name, value in losses.items()}
    result["loss"] = float(loss.detach().item())
    return result


def save_checkpoint(
    path: Path,
    model: TransformerPolicyValue | CNNPolicyValue,
    model_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
        },
        path,
    )


def train_streaming(
    input_path: Path,
    checkpoint_out: Path,
    checkpoint_in: Path | None,
    max_positions: int,
    min_depth: int,
    multipv: int,
    cp_scale: float,
    policy_temperature: float,
    batch_size: int,
    train_steps: int | None,
    passes: int,
    lr: float,
    value_weight: float,
    material_weight: float,
    mate_in_1_weight: float,
    in_check_weight: float,
    has_check_weight: float,
    capture_available_weight: float,
    legal_mobility_weight: float,
    attack_pressure_weight: float,
    king_pressure_weight: float,
    best_capture_weight: float,
    hanging_material_weight: float,
    model_type: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    ff_dim: int,
    dropout: float,
    cnn_channels: int,
    cnn_blocks: int,
    cnn_dropout: float,
    sparse_policy_target: bool,
    trust_stockfish_legal_moves: bool,
    skip_positions: int,
    amp: bool,
    amp_no_scaler: bool,
    tf32: bool,
    device_name: str,
    log_every: int,
    save_every: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if passes <= 0:
        raise ValueError("passes must be positive.")

    if train_steps is not None and train_steps <= 0:
        raise ValueError("train_steps must be positive when provided.")

    if skip_positions < 0:
        raise ValueError("skip_positions must be non-negative.")

    device = torch.device(device_name)
    model = make_model(
        checkpoint_in=checkpoint_in,
        model_type=model_type,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        cnn_channels=cnn_channels,
        cnn_blocks=cnn_blocks,
        cnn_dropout=cnn_dropout,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    amp_enabled = bool(amp and device.type == "cuda")
    amp_scaler_enabled = bool(amp_enabled and not amp_no_scaler)
    tf32_enabled = configure_tf32(tf32=tf32, device=device)
    scaler = make_amp_scaler(device=device, amp=amp_enabled, amp_no_scaler=amp_no_scaler)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    normalized_model_type = model_type.lower()
    if normalized_model_type == "transformer":
        model_config = {
            "model_type": "transformer",
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ff_dim": ff_dim,
            "dropout": dropout,
        }
    elif normalized_model_type == "cnn":
        model_config = {
            "model_type": "cnn",
            "channels": cnn_channels,
            "n_blocks": cnn_blocks,
            "dropout": cnn_dropout,
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    
    compute_aux_targets = aux_heads_enabled(
        material_weight=material_weight,
        mate_in_1_weight=mate_in_1_weight,
        in_check_weight=in_check_weight,
        has_check_weight=has_check_weight,
        capture_available_weight=capture_available_weight,
        legal_mobility_weight=legal_mobility_weight,
        attack_pressure_weight=attack_pressure_weight,
        king_pressure_weight=king_pressure_weight,
        best_capture_weight=best_capture_weight,
        hanging_material_weight=hanging_material_weight,
    )

    step = 0
    total_examples = 0

    print()
    print("Streaming Lichess eval training")
    print("----------------------------------------")
    print(f"Input path:       {input_path}")
    print(f"Checkpoint in:    {checkpoint_in if checkpoint_in is not None else 'fresh random model'}")
    print(f"Checkpoint out:   {checkpoint_out}")
    print(f"Max positions/pass: {max_positions}")
    print(f"Skip positions:       {skip_positions}")
    print(f"Passes:           {passes}")
    print(f"Train steps:      {train_steps if train_steps is not None else 'until data exhausted'}")
    print(f"Min depth:        {min_depth}")
    print(f"MultiPV:          {multipv}")
    print(f"Batch size:       {batch_size}")
    print(f"LR:               {lr}")
    print(f"Value weight:     {value_weight}")
    print(f"Material weight:  {material_weight}")
    print(f"Mate-in-1 weight: {mate_in_1_weight}")
    print(f"In-check weight:  {in_check_weight}")
    print(f"Has-check weight: {has_check_weight}")
    print(f"Capture weight:   {capture_available_weight}")
    print(f"Mobility weight:  {legal_mobility_weight}")
    print(f"Attack pressure:  {attack_pressure_weight}")
    print(f"King pressure:    {king_pressure_weight}")
    print(f"Best capture:     {best_capture_weight}")
    print(f"Hanging material: {hanging_material_weight}")
    print(f"Compute aux labels: {compute_aux_targets}")
    print(f"Sparse policy target: {sparse_policy_target}")
    print(f"Trust Stockfish legal moves: {trust_stockfish_legal_moves}")
    print(f"AMP mixed precision: {amp_enabled}")
    print(f"AMP GradScaler:     {amp_scaler_enabled}")
    print(f"TF32 enabled:       {tf32_enabled}")
    print(f"Fast FEN stream path: {sparse_policy_target and trust_stockfish_legal_moves and not compute_aux_targets}")
    print(f"Model type:       {normalized_model_type}")
    if normalized_model_type == "transformer":
        print(f"Model config:     d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, ff_dim={ff_dim}, dropout={dropout}")
    else:
        print(f"Model config:     channels={cnn_channels}, n_blocks={cnn_blocks}, dropout={cnn_dropout}")
    print(f"Device:           {device}")
    print()

    for pass_idx in range(1, passes + 1):
        batch_examples: list[dict[str, Any]] = []
        last_scanned = 0
        pass_examples = 0

        for example in iter_training_examples(
            input_path=input_path,
            max_positions=max_positions,
            min_depth=min_depth,
            multipv=multipv,
            cp_scale=cp_scale,
            policy_temperature=policy_temperature,
            compute_aux_targets=compute_aux_targets,
            sparse_policy_target=sparse_policy_target,
            trust_stockfish_legal_moves=trust_stockfish_legal_moves,
            skip_positions=skip_positions,
        ):
            batch_examples.append(example)
            scanned_value = example["scanned"]
            last_scanned = int(scanned_value.item()) if hasattr(scanned_value, "item") else int(scanned_value)

            if len(batch_examples) < batch_size:
                continue

            batch = stack_batch(batch_examples, device=device, sparse_policy_target=sparse_policy_target)
            batch_examples.clear()

            step += 1
            total_examples += batch_size
            pass_examples += batch_size

            should_log = step == 1 or (log_every > 0 and step % log_every == 0)
            metrics = train_batch(
                model=model,
                optimizer=optimizer,
                batch=batch,
                value_weight=value_weight,
                sparse_policy_target=sparse_policy_target,
                material_weight=material_weight,
                mate_in_1_weight=mate_in_1_weight,
                in_check_weight=in_check_weight,
                has_check_weight=has_check_weight,
                capture_available_weight=capture_available_weight,
                legal_mobility_weight=legal_mobility_weight,
                attack_pressure_weight=attack_pressure_weight,
                king_pressure_weight=king_pressure_weight,
                best_capture_weight=best_capture_weight,
                hanging_material_weight=hanging_material_weight,
                return_metrics=should_log,
                amp=amp_enabled,
                amp_no_scaler=amp_no_scaler,
                scaler=scaler,
            )

            if step == 1 or (log_every > 0 and step % log_every == 0):
                print(
                    f"pass={pass_idx:02d} "
                    f"step={step:05d} "
                    f"examples={total_examples} "
                    f"pass_examples={pass_examples} "
                    f"scanned={last_scanned} "
                    f"loss={metrics['loss']:.6f} "
                    f"policy={metrics['policy']:.6f} "
                    f"value={metrics['value']:.6f} "
                    f"material={metrics['material']:.6f} "
                    f"mate_in_1={metrics['mate_in_1']:.6f} "
                    f"in_check={metrics['in_check']:.6f} "
                    f"has_check={metrics['has_check']:.6f} "
                    f"capture={metrics['capture_available']:.6f} "
                    f"mobility={metrics['legal_mobility']:.6f} "
                    f"attack={metrics['attack_pressure']:.6f} "
                    f"king={metrics['king_pressure']:.6f} "
                    f"bestcap={metrics['best_capture']:.6f} "
                    f"hanging={metrics['hanging_material']:.6f}"
                )

            if save_every > 0 and step % save_every == 0:
                save_checkpoint(checkpoint_out, model, model_config=model_config)
                print(f"Saved checkpoint: {checkpoint_out}")

            if train_steps is not None and step >= train_steps:
                save_checkpoint(checkpoint_out, model, model_config=model_config)
                print()
                print(f"Done. examples={total_examples} steps={step}")
                print(f"Saved checkpoint: {checkpoint_out}")
                return

        if batch_examples and (train_steps is None or step < train_steps):
            batch = stack_batch(batch_examples, device=device, sparse_policy_target=sparse_policy_target)
            step += 1
            total_examples += len(batch_examples)
            pass_examples += len(batch_examples)

            metrics = train_batch(
                model=model,
                optimizer=optimizer,
                batch=batch,
                value_weight=value_weight,
                sparse_policy_target=sparse_policy_target,
                material_weight=material_weight,
                mate_in_1_weight=mate_in_1_weight,
                in_check_weight=in_check_weight,
                has_check_weight=has_check_weight,
                capture_available_weight=capture_available_weight,
                legal_mobility_weight=legal_mobility_weight,
                attack_pressure_weight=attack_pressure_weight,
                king_pressure_weight=king_pressure_weight,
                best_capture_weight=best_capture_weight,
                hanging_material_weight=hanging_material_weight,
                amp=amp_enabled,
                amp_no_scaler=amp_no_scaler,
                scaler=scaler,
            )

            print(
                f"pass={pass_idx:02d} "
                f"step={step:05d} "
                f"examples={total_examples} "
                f"pass_examples={pass_examples} "
                f"scanned={last_scanned} "
                f"loss={metrics['loss']:.6f} "
                f"policy={metrics['policy']:.6f} "
                f"value={metrics['value']:.6f} "
                f"material={metrics['material']:.6f} "
                f"mate_in_1={metrics['mate_in_1']:.6f} "
                f"in_check={metrics['in_check']:.6f} "
                f"has_check={metrics['has_check']:.6f} "
                f"capture={metrics['capture_available']:.6f} "
                f"mobility={metrics['legal_mobility']:.6f} "
                f"attack={metrics['attack_pressure']:.6f} "
                f"king={metrics['king_pressure']:.6f} "
                f"bestcap={metrics['best_capture']:.6f} "
                f"hanging={metrics['hanging_material']:.6f}"
            )

        print(f"Completed pass {pass_idx}: pass_examples={pass_examples}")

    save_checkpoint(checkpoint_out, model, model_config=model_config)
    print()
    print(f"Done. examples={total_examples} steps={step}")
    print(f"Saved checkpoint: {checkpoint_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Lichess Stockfish eval JSONL/JSONL.ZST directly into transformer training."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--max-positions", type=int, default=100_000)
    parser.add_argument(
        "--skip-positions",
        type=int,
        default=0,
        help="Skip this many accepted lines before parsing/training. Useful for continuing on a later slice of a large JSONL/ZST stream.",
    )
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--policy-temperature", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=2.0)
    parser.add_argument("--material-weight", type=float, default=0.0)
    parser.add_argument("--mate-in-1-weight", type=float, default=0.0)
    parser.add_argument("--in-check-weight", type=float, default=0.0)
    parser.add_argument("--has-check-weight", type=float, default=0.0)
    parser.add_argument("--capture-available-weight", type=float, default=0.0)
    parser.add_argument("--legal-mobility-weight", type=float, default=0.0)
    parser.add_argument("--attack-pressure-weight", type=float, default=0.0)
    parser.add_argument("--king-pressure-weight", type=float, default=0.0)
    parser.add_argument("--best-capture-weight", type=float, default=0.0)
    parser.add_argument("--hanging-material-weight", type=float, default=0.0)
    parser.add_argument("--model-type", choices=["transformer", "cnn"], default=MODEL_TYPE)
    parser.add_argument("--d-model", type=int, default=MODEL_CONFIG["d_model"])
    parser.add_argument("--n-layers", type=int, default=MODEL_CONFIG["n_layers"])
    parser.add_argument("--n-heads", type=int, default=MODEL_CONFIG["n_heads"])
    parser.add_argument("--ff-dim", type=int, default=MODEL_CONFIG["ff_dim"])
    parser.add_argument("--dropout", type=float, default=MODEL_CONFIG["dropout"])
    parser.add_argument("--cnn-channels", type=int, default=CNN_CONFIG["channels"])
    parser.add_argument("--cnn-blocks", type=int, default=CNN_CONFIG["n_blocks"])
    parser.add_argument("--cnn-dropout", type=float, default=CNN_CONFIG["dropout"])
    parser.add_argument(
        "--sparse-policy-target",
        action="store_true",
        help="Use compact Stockfish PV targets instead of dense legal-masked policy targets.",
    )
    parser.add_argument(
        "--trust-stockfish-legal-moves",
        action="store_true",
        help="Skip legal move validation for Stockfish PV moves. Faster but assumes input data is valid.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP mixed precision training.")
    parser.add_argument(
        "--amp-no-scaler",
        action="store_true",
        help="Use autocast without GradScaler. Faster on some GPUs, but less numerically safe.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul/cuDNN kernels on CUDA for faster FP32 conv/matmul.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_streaming(
        input_path=args.input_path,
        checkpoint_out=args.checkpoint_out,
        checkpoint_in=args.checkpoint_in,
        max_positions=args.max_positions,
        min_depth=args.min_depth,
        multipv=args.multipv,
        cp_scale=args.cp_scale,
        policy_temperature=args.policy_temperature,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        passes=args.passes,
        lr=args.lr,
        value_weight=args.value_weight,
        material_weight=args.material_weight,
        mate_in_1_weight=args.mate_in_1_weight,
        in_check_weight=args.in_check_weight,
        has_check_weight=args.has_check_weight,
        capture_available_weight=args.capture_available_weight,
        legal_mobility_weight=args.legal_mobility_weight,
        attack_pressure_weight=args.attack_pressure_weight,
        king_pressure_weight=args.king_pressure_weight,
        best_capture_weight=args.best_capture_weight,
        hanging_material_weight=args.hanging_material_weight,
        model_type=args.model_type,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
        cnn_blocks=args.cnn_blocks,
        cnn_dropout=args.cnn_dropout,
        sparse_policy_target=args.sparse_policy_target,
        skip_positions=args.skip_positions,
        amp=args.amp,
        amp_no_scaler=args.amp_no_scaler,
        tf32=args.tf32,
        device_name=args.device,
        log_every=args.log_every,
        save_every=args.save_every,
        trust_stockfish_legal_moves=args.trust_stockfish_legal_moves,
    )


if __name__ == "__main__":
    main()