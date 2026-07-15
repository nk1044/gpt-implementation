import torch
import logging
import os

from model import LanguageModel
from data import get_batch
from config import (
    N_TRAIN, LEARNING_RATE,
    LOG_INTERVAL, EVAL_INTERVAL, EVAL_ITERS,
    CHECKPOINT_DIR, CHECKPOINT_INTERVAL,
    DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        split_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model = LanguageModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
logger.info(f"Model initialized: {n_params:.2f}M parameters")
logger.info(f"Device: {DEVICE}")
logger.info(f"Training for {N_TRAIN} steps | lr={LEARNING_RATE} | eval every {EVAL_INTERVAL} steps")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for step in range(N_TRAIN):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        logger.info(
            f"Step {step:>5}/{N_TRAIN} | "
            f"train loss: {losses['train']:.4f} | "
            f"val loss: {losses['val']:.4f}"
        )

    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % LOG_INTERVAL == 0:
        logger.info(f"Step {step:>5}/{N_TRAIN} | step loss: {loss.item():.4f}")

    if step % CHECKPOINT_INTERVAL == 0 and step > 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}.pt")
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved: {ckpt_path}")

losses = estimate_loss(model)
logger.info(
    f"Training complete | "
    f"train loss: {losses['train']:.4f} | "
    f"val loss: {losses['val']:.4f}"
)

final_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
torch.save(
    {
        "step": N_TRAIN,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses["val"],
    },
    final_path,
)
logger.info(f"Final model saved: {final_path}")
