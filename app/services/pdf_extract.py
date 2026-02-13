from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class ExtractedPage:
    page_number: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> List[ExtractedPage]:
    """Extract text from a Unicode-text PDF. (No OCR.)

    Returns list of ExtractedPage with 1-based page numbers.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    doc = fitz.open(str(pdf_path))
    pages: List[ExtractedPage] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # Light normalization (keep Amharic chars)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join([line.rstrip() for line in text.split("\n")]).strip()
        pages.append(ExtractedPage(page_number=i + 1, text=text))
    doc.close()
    return pages
