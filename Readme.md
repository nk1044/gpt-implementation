# GPT Implementation

A from-scratch character-level GPT trained on plain text, built with PyTorch.

## Architecture

- Transformer decoder with multi-head causal self-attention
- Learned token + positional embeddings
- Layer norm, feed-forward blocks, dropout regularization
- Default: 6 layers, 8 heads, 128-dim embeddings (~3M parameters)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch
```

Place your training corpus as `pg3200.txt` in the project root.

## Training

```bash
python main.py
```

- Logs step loss every 100 steps and train/val loss every 500 steps to stdout and `training.log`
- Saves checkpoints to `checkpoints/checkpoint_step_N.pt` every 1000 steps
- Saves final model to `checkpoints/model_final.pt`
- Automatically uses CUDA → MPS → CPU

Tune hyperparameters in [config.py](config.py).

## Inference

```bash
python test.py
```

Loads the latest checkpoint and enters an interactive loop:

```
Enter max tokens to generate (0 to exit): 200
--- Generated Text ---
...
--- End (200 tokens) ---

Enter max tokens to generate (0 to exit): 0
Exiting.
```

## Files

| File | Description |
|---|---|
| `config.py` | All hyperparameters and device selection |
| `tokenizer.py` | Character-level tokenizer (encode/decode) |
| `data.py` | Batch sampler for train/val splits |
| `attention_head.py` | Single attention head and multi-head attention |
| `transformer_block.py` | Transformer block (attention + feed-forward + layer norm) |
| `model.py` | Full language model with generate() |
| `main.py` | Training loop with logging and checkpointing |
| `test.py` | Interactive inference from a saved checkpoint |
