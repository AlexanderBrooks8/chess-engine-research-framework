# Training and Data Pipeline

## Self-Play

The self-play layer records board states, policy targets, and final outcomes into reusable training examples. The MCTS worker can generate visit-count policy targets directly from search.

`generate_mcts_selfplay.py` provides a compact entry point for producing MCTS self-play records, while `train_iterative_mcts_selfplay.py` demonstrates an iterative generate/train workflow.

## External Evaluation Data

`train_lichess_eval_stream.py` is a streaming training pipeline for externally supplied Lichess evaluation data. The public repository contains the trainer, not the dataset itself.

The script supports:

- Transformer or residual-CNN models
- checkpoint initialization and checkpoint output
- configurable batch size and learning rate
- value and auxiliary-task weights
- CUDA AMP mixed-precision training
- position-count limits and streaming passes

Some stream-processing paths use NumPy and Zstandard for efficient handling of compressed evaluation data.

## Objectives

The core training utilities expose policy cross-entropy and value mean-squared error losses. The model architectures additionally expose auxiliary chess-state outputs that can be weighted by the streaming trainer.

## Checkpoints

Checkpoint helpers store model and optimizer state plus model configuration. Checkpoint files are intentionally excluded from the public repository because they are generated artifacts and can be large.
