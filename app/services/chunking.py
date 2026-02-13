from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ChunkResult:
    chunk_index: int
    chunk_text: str


def _split_paragraphs(text: str) -> List[str]:
    # Split by blank lines; fallback to single lines if needed
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def chunk_text(text: str, target_chars: int = 1000, overlap_ratio: float = 0.15) -> List[ChunkResult]:
    """Chunk text into roughly target_chars with overlap.

    Overlap is applied as a tail of the previous chunk that is prefixed to the next chunk.
    """
    text = (text or "").strip()
    if not text:
        return []

    overlap_chars = max(0, int(target_chars * overlap_ratio))
    paras = _split_paragraphs(text)

    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paras:
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= target_chars:
            buf = buf + "\n\n" + para
        else:
            flush()
            buf = para

    flush()

    # Apply overlap by carrying tail forward
    out: List[ChunkResult] = []
    prev_tail = ""
    for idx, c in enumerate(chunks):
        if prev_tail:
            combined = (prev_tail + c).strip()
        else:
            combined = c.strip()
        out.append(ChunkResult(chunk_index=idx, chunk_text=combined))
        prev_tail = combined[-overlap_chars:] if overlap_chars > 0 and len(combined) > overlap_chars else ""

    return out
