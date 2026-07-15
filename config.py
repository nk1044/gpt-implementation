import torch

# Model architecture
CONTEXT_WINDOW = 128     # (T) - doubled for more context
BATCH_SIZE = 32          # (B) - larger batches for stable gradients
N_EMBEDDINGS = 128       # (C) - richer token representations
NUM_LAYERS = 6           # balanced depth for this embedding size
FFN_FACTOR = 4
NUM_HEADS = 8

# Training
N_TRAIN = 10000
LEARNING_RATE = 3e-4     # standard AdamW learning rate
DROPOUTS = 0.2           # regularization

# Logging
LOG_INTERVAL = 100       # log step loss every N steps
EVAL_INTERVAL = 500      # run loss estimation every N steps
EVAL_ITERS = 200         # number of batches to average for loss estimation

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 1000   # save checkpoint every N steps

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
