# Architecture

## Shared Engine Interface

All playable engines implement a shared `BaseEngine` contract with a `choose_move(board)` method. This allows the arena, self-play workers, CLI, and integration layers to swap engine implementations without changing orchestration logic.

## Position Encoding

A chess position is represented by:

- 64 piece tokens (`0` empty, `1-6` white pieces, `7-12` black pieces)
- side to move
- four castling-right flags packed into a 0-15 value
- en-passant square or a sentinel value

This same representation feeds both the Transformer and CNN policy-value networks.

## Policy Action Space

Moves are represented by a fixed UCI vocabulary generated from source/destination square pairs and promotion variants. The resulting action space contains 4,544 actions. A legal-move mask restricts policy evaluation to moves legal in the current board state.

## Classical Search

The minimax engine uses negamax with alpha-beta pruning. Static evaluation is based on side-to-move material balance plus a small mobility term.

## Neural Policy-Value Models

### Transformer

The Transformer embeds piece identities, square positions, side-to-move, castling rights, and en-passant state. A learned classification token is processed with the board tokens through the encoder and is then used for policy/value prediction.

Default configuration:

- `d_model = 192`
- `n_layers = 4`
- `n_heads = 6`
- `ff_dim = 768`
- `dropout = 0.1`

The network also exposes auxiliary chess-state heads used by training experiments.

### Residual CNN

The CNN uses an 8x8 feature map with piece embeddings and state metadata, followed by a residual tower.

Default configuration:

- `channels = 128`
- `n_blocks = 6`
- `dropout = 0.1`

It exposes the same policy-value interface as the Transformer so the model factory and downstream engine code can remain architecture-independent.

## PUCT MCTS

The MCTS implementation follows a policy-prior/value-evaluation workflow:

1. Encode and evaluate the root position.
2. Expand legal children using neural policy priors.
3. Select children with a PUCT score.
4. Evaluate newly reached leaves with the policy-value model.
5. Back up values through the search path.
6. Build a policy target from root visit counts.
7. Select a move from visit counts using the configured temperature.

Additional implementation features include batched leaf evaluation, an evaluation cache, configurable maximum child count, simulation/time budgets, root Dirichlet noise, and search statistics.

## Tactical Veto

The MCTS engine can optionally pass its highest-visit candidates through a small classical tactical search. This provides an additional guard against tactically poor moves before final selection.

## Self-Play and Training

Self-play workers produce `GameRecord` objects containing positions, selected policies, and results. These are converted into training examples and fed through policy/value losses.

The repository also contains a streaming trainer for externally generated Lichess position/evaluation data.

## Arena

The arena treats engines through the common interface and handles:

- alternating colors
- optional opening FENs
- PGN construction
- maximum-ply limits
- game-result aggregation
- optional material adjudication

This provides a common evaluation harness for classical, direct-neural, and MCTS engines.
