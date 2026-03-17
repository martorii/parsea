from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """
    Coordinates in PDF points (origin = bottom-left for pypdf,
    top-left for pdfplumber — we normalise to pdfplumber convention).
    x0, y0 = top-left corner
    x1, y1 = bottom-right corner
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class TextBlock:
    text: str
    bbox: BoundingBox


@dataclass
class Table:
    rows: list[list[Any]]  # list of rows; each row is a list of cell values
    bbox: BoundingBox


@dataclass
class Picture:
    index: int  # image index on the page
    width: float
    height: float
    bbox: BoundingBox
    image_bytes: bytes | None = None  # raw image bytes (None if unavailable)


@dataclass
class ParsedPage:
    page_number: int  # 1-based
    text_blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""
    tables: list[Table] = field(default_factory=list)
    pictures: list[Picture] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "pages": [
                {
                    "page_number": page.page_number,
                    "full_text": page.full_text,
                    # "text_blocks": [
                    #     {"text": tb.text, "bbox": tb.bbox.as_dict()}
                    #     for tb in page.text_blocks
                    # ],
                    "tables": [
                        {"rows": t.rows, "bbox": t.bbox.as_dict()} for t in page.tables
                    ],
                    # "pictures": [
                    #     {
                    #         "index": p.index,
                    #         "width": p.width,
                    #         "height": p.height,
                    #         "bbox": p.bbox.as_dict(),
                    #         "has_bytes": p.image_bytes is not None,
                    #     }
                    #     for p in page.pictures
                    # ],
                }
                for page in self.pages
            ],
        }
