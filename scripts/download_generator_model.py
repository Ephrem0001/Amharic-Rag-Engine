"""Pre-download the RAG generator model so /rag/ask is fast. Shows progress in the terminal."""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Show progress bars (disable = 0 means show)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "rasyosef/Llama-3.2-400M-Amharic"
    print(f"Downloading tokenizer and model: {model_id}")
    print("(Progress bars appear below. First run may take several minutes.)\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer downloaded.\n")

    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("\nModel downloaded. Cache is ready for POST /rag/ask.")


if __name__ == "__main__":
    main()

