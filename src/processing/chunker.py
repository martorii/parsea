from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.parser import ParsedPage
from utils import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    id: str
    page: int
    index: int
    text: str

    @property
    def preview(self) -> str:
        return self.text[:120] + ("…" if len(self.text) > 120 else self.text)


def _make_id(page: int, index: int, text: str) -> str:
    raw = f"{page}:{index}:{text[:40]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = 600,
    overlap: int = 120,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Chunk] = []

    for page_content in pages:
        texts = splitter.split_text(page_content.full_text)
        log.debug(
            "Page %d (%d chars) → %d chunks",
            page_content.page_number,
            len(page_content.full_text),
            len(texts),
        )
        for idx, text in enumerate(texts):
            all_chunks.append(
                Chunk(
                    id=_make_id(page_content.page_number, idx, text),
                    page=page_content.page_number,
                    index=idx,
                    text=text,
                )
            )

    log.info(
        "Chunker: %d pages → %d chunks (size=%d, overlap=%d)",
        len(pages),
        len(all_chunks),
        chunk_size,
        overlap,
    )
    return all_chunks
