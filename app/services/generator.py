from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings


def _parse_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "").lower().strip()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=1)
def _load_generator():
    model_id = settings.GENERATOR_MODEL
    dtype = _parse_dtype(settings.TORCH_DTYPE)
    device = settings.DEVICE

    logger.info(f"Loading generator model: {model_id} (device={device}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use accelerate if available; on CPU this is okay.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()
    return tokenizer, model


def build_prompt(question: str, contexts: List[Tuple[int, str]]) -> str:
    """contexts: list of (page_number, chunk_text)."""
    ctx_lines = []
    for i, (page, txt) in enumerate(contexts, start=1):
        snippet = (txt or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200].rstrip() + "..."
        ctx_lines.append(f"[{i}] (ገጽ {page})\n{snippet}")

    ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(ምንም ኮንቴክስት አልተገኘም)"

    prompt = f"""አንተ በአማርኛ የሚመልስ ረዳት ነህ።
ከታች ባለው ኮንቴክስት ብቻ ተጠቅመህ መልስ ስጥ።
መረጃው በኮንቴክስት ውስጥ ካልተገኘ በግልፅ “በተሰጠው ይዘት ውስጥ መልስ አልተገኘም።” በማለት መልስ ስጥ።
መልስህ ውስጥ የተጠቀምክባቸውን ምንጮች በ[1], [2] እና እንዲሁ ምልክቶች አሳይ።

### ኮንቴክስት
{ctx_block}

### ጥያቄ
{question}

### መልስ (በአማርኛ)
"""
    return prompt


def generate_answer(question: str, contexts: List[Tuple[int, str]]) -> str:
    tokenizer, model = _load_generator()
    prompt = build_prompt(question, contexts)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(settings.TEXTGEN_MAX_NEW_TOKENS),
            do_sample=True,
            temperature=float(settings.TEXTGEN_TEMPERATURE),
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Return text after the "### መልስ" marker if present
    marker = "### መልስ"
    if marker in decoded:
        decoded = decoded.split(marker, 1)[-1]
    return decoded.strip()
