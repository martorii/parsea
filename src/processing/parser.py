from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from typing import Any

import pdfplumber
from pypdf import PdfReader

from models.parser import (
    BoundingBox,
    ParsedDocument,
    ParsedPage,
    Picture,
    Table,
    TextBlock,
)
from utils import get_logger

log = get_logger(__name__)


# ─── Abstract base ────────────────────────────────────────────────────────────


class DocumentParser(ABC):
    """
    Base class for all document format parsers.

    Subclasses declare `extensions` as class-level sets, then
    implement `parse(data) -> ParsedDocument`.
    """

    extensions: set[str] = set()
    label: str = "unknown"

    def supports(self, filename: str = "") -> bool:
        if filename:
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext in self.extensions:
                return True
        return False

    @abstractmethod
    def parse(self, data: bytes) -> ParsedDocument:
        return ParsedDocument()

    def __repr__(self) -> str:
        return f"<DocumentParser:{self.label}>"


# ─── PDF ─────────────────────────────────────────────────────────────────────


class PdfParser(DocumentParser):
    extensions = {"pdf"}
    label = "PDF"

    def parse(self, data: bytes) -> ParsedDocument:
        """
        Parse a PDF document from bytes and return a :class:`ParsedDocument`.

        Parameters
        ----------
        data:
            PDF file data as bytes.

        Returns
        -------
        ParsedDocument
            Structured extraction result containing text, tables and pictures
            together with their bounding-box coordinates for every page.
        """

        # ------------------------------------------------------------------
        # Open with both libraries using BytesIO:
        # ------------------------------------------------------------------
        log.info("Opening document with pdfplumber …")
        plumber_pdf = pdfplumber.open(io.BytesIO(data))

        log.info("Opening document with pypdf …")
        pypdf_reader = PdfReader(io.BytesIO(data))

        total_pages = len(plumber_pdf.pages)
        log.info("Total pages detected: %d", total_pages)

        document = ParsedDocument()

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            log.info("── Processing page %d / %d ──", page_number, total_pages)

            plumber_page = plumber_pdf.pages[page_idx]
            pypdf_page = pypdf_reader.pages[page_idx]

            parsed_page = ParsedPage(page_number=page_number)

            # --------------------------------------------------------------
            # 1. TEXT EXTRACTION
            # --------------------------------------------------------------
            parsed_page = self._extract_text(plumber_page, parsed_page)

            # --------------------------------------------------------------
            # 2. TABLE EXTRACTION
            # --------------------------------------------------------------
            parsed_page = self._extract_tables(plumber_page, parsed_page)

            # --------------------------------------------------------------
            # 3. PICTURE / IMAGE EXTRACTION
            # --------------------------------------------------------------
            parsed_page = self._extract_pictures(plumber_page, pypdf_page, parsed_page)

            document.pages.append(parsed_page)
            log.info(
                "Page %d done — text_blocks=%d, tables=%d, pictures=%d",
                page_number,
                len(parsed_page.text_blocks),
                len(parsed_page.tables),
                len(parsed_page.pictures),
            )

        plumber_pdf.close()
        log.info("Parsing complete. Pages processed: %d", total_pages)
        return document

    def _extract_text(self, plumber_page: Any, parsed_page: ParsedPage) -> ParsedPage:
        """Extract word-level text blocks with bounding boxes."""
        log.debug("  [text] Extracting words from page %d …", parsed_page.page_number)

        words = plumber_page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        )

        if not words:
            log.debug("  [text] No words found on page %d.", parsed_page.page_number)
            parsed_page.full_text = ""
            return parsed_page

        for word in words:
            bbox = BoundingBox(
                x0=word["x0"],
                y0=word["top"],  # pdfplumber uses 'top' (distance from top edge)
                x1=word["x1"],
                y1=word["bottom"],
            )
            parsed_page.text_blocks.append(TextBlock(text=word["text"], bbox=bbox))

        # Full plain text (preserving layout)
        parsed_page.full_text = plumber_page.extract_text() or ""
        log.debug(
            "  [text] Extracted %d word blocks (%d chars total).",
            len(parsed_page.text_blocks),
            len(parsed_page.full_text),
        )
        return parsed_page

    def _extract_tables(self, plumber_page: Any, parsed_page: ParsedPage) -> ParsedPage:
        """Extract tables together with their bounding boxes."""
        log.debug(
            "  [tables] Extracting tables from page %d …", parsed_page.page_number
        )

        # find_tables() returns Table objects with .bbox and .extract()
        plumber_tables = plumber_page.find_tables()

        if not plumber_tables:
            log.debug("  [tables] No tables found on page %d.", parsed_page.page_number)
            return parsed_page

        for t_idx, plumber_table in enumerate(plumber_tables):
            raw_bbox = plumber_table.bbox  # (x0, top, x1, bottom)
            bbox = BoundingBox(
                x0=raw_bbox[0],
                y0=raw_bbox[1],
                x1=raw_bbox[2],
                y1=raw_bbox[3],
            )
            rows = plumber_table.extract()  # list[list[str | None]]
            parsed_page.tables.append(Table(rows=rows, bbox=bbox))
            log.debug(
                "  [tables] Table %d: %d rows, bbox=%s",
                t_idx + 1,
                len(rows),
                bbox.as_dict(),
            )

        log.debug(
            "  [tables] Total tables on page %d: %d",
            parsed_page.page_number,
            len(parsed_page.tables),
        )
        return parsed_page

    def _extract_pictures(
        self,
        plumber_page: Any,
        pypdf_page: Any,
        parsed_page: ParsedPage,
    ) -> ParsedPage:
        """
        Extract images and their bounding boxes.

        pdfplumber exposes image metadata (coordinates, width, height).
        pypdf is used to pull the actual image bytes.
        """
        log.debug(
            "  [pictures] Extracting images from page %d …", parsed_page.page_number
        )

        # pdfplumber stores image metadata under .images
        plumber_images = plumber_page.images
        if not plumber_images:
            log.debug(
                "  [pictures] No images found on page %d.", parsed_page.page_number
            )
            return parsed_page

        # Build image-bytes list from pypdf (order matches pdfplumber order)
        pypdf_images: list[Any] = []
        try:
            pypdf_images = list(pypdf_page.images)
        except Exception as exc:
            log.warning("  [pictures] Could not extract image bytes via pypdf: %s", exc)

        for img_idx, img_meta in enumerate(plumber_images):
            bbox = BoundingBox(
                x0=img_meta.get("x0", 0.0),
                y0=img_meta.get("top", 0.0),  # pdfplumber convention
                x1=img_meta.get("x1", 0.0),
                y1=img_meta.get("bottom", 0.0),
            )

            # Attempt to retrieve raw bytes from pypdf
            img_bytes: bytes | None = None
            if img_idx < len(pypdf_images):
                try:
                    img_bytes = pypdf_images[img_idx].data
                except Exception as exc:
                    log.debug(
                        "  [pictures] Could not read bytes for image %d: %s",
                        img_idx,
                        exc,
                    )

            picture = Picture(
                index=img_idx,
                width=img_meta.get("width", 0.0),
                height=img_meta.get("height", 0.0),
                bbox=bbox,
                image_bytes=img_bytes,
            )
            parsed_page.pictures.append(picture)
            log.debug(
                "  [pictures] Image %d: size=(%s x %s), bbox=%s, bytes=%s",
                img_idx,
                picture.width,
                picture.height,
                bbox.as_dict(),
                f"{len(img_bytes)} bytes" if img_bytes else "unavailable",
            )

        log.debug(
            "  [pictures] Total images on page %d: %d",
            parsed_page.page_number,
            len(parsed_page.pictures),
        )
        return parsed_page


# ─── Registry & dispatcher ────────────────────────────────────────────────────

_PARSERS: list[DocumentParser] = []


def _register(cls: type[DocumentParser]) -> type[DocumentParser]:
    _PARSERS.append(cls())
    return cls


for _cls in [PdfParser]:
    _register(_cls)


def get_parser(filename: str = "") -> DocumentParser | None:
    """Return the first parser that claims to support the given file."""
    for parser in _PARSERS:
        if parser.supports(filename=filename):
            return parser
    return None


def parse_document(data: bytes, filename: str) -> ParsedDocument:
    """
    Dispatch to the correct DocumentParser subclass and return parsed pages.
    Raises ValueError for unsupported formats.
    """
    parser = get_parser(filename=filename)
    if parser is None:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "?"
        supported = sorted({e for p in _PARSERS for e in p.extensions})
        raise ValueError(
            f"Unsupported file type: .{ext}. "
            f"Supported extensions: {', '.join(supported)}"
        )
    log.info("Parsing %r with %s", filename, parser)
    return parser.parse(data)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
