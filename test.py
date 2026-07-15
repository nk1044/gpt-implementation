import os
import torch

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

        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=max_tokens)

        output = decode(generated[0].tolist())
        print("\n--- Generated Text ---")
        print(output)
        print(f"--- End ({len(output)} chars, {max_tokens} tokens) ---\n")


if __name__ == "__main__":
    main()
