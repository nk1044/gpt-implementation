import os
import sys
import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import decode
from config import CHECKPOINT_DIR, CONTEXT_WINDOW, DEVICE


def load_model(checkpoint_path: str) -> LanguageModel:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = LanguageModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def find_checkpoint() -> str:
    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
    if os.path.exists(final_path):
        return final_path

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(
            f"Checkpoint directory '{CHECKPOINT_DIR}' not found. Run main.py first."
        )

    candidates = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files found in '{CHECKPOINT_DIR}'. Run main.py first."
        )
    return os.path.join(CHECKPOINT_DIR, candidates[-1])


def main():
    checkpoint_path = find_checkpoint()
    print(f"Loading model from: {checkpoint_path}")

    model = load_model(checkpoint_path)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {n_params:.2f}M parameters")
    print(f"Context window: {CONTEXT_WINDOW} tokens")
    print()

    while True:
        try:
            raw = input("Enter max tokens to generate (0 to exit): ").strip()
            max_tokens = int(raw)
        except (ValueError, EOFError):
            print("Please enter a valid integer.\n")
            continue

        if max_tokens == 0:
            print("Exiting.")
            break

        print("\n--- Generated Text ---")
        x = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            for _ in range(max_tokens):
                x_cond = x[:, -CONTEXT_WINDOW:]
                logits, _ = model(x_cond)
                logits = logits[:, -1, :]
                prob = F.softmax(logits, dim=-1)
                x_next = torch.multinomial(prob, num_samples=1)
                x = torch.cat((x, x_next), dim=-1)
                sys.stdout.write(decode([x_next.item()]))
                sys.stdout.flush()
        print(f"\n--- End ({max_tokens} tokens) ---\n")


if __name__ == "__main__":
    main()
