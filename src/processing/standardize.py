from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import get_close_matches
from typing import Any

from models import FieldDefinition, FieldType
from utils import get_logger

log = get_logger(__name__)

# Return type for every standardize() call
StdResult = tuple[Any, str | None]  # (value, warning | None)


# ─── Abstract base ────────────────────────────────────────────────────────────


class DataType(ABC):
    """
    Base class for all field type standardizers.

    Subclasses must implement `standardize(raw, field_def)`.
    They may optionally override `output_description` for documentation.
    """

    field_type: FieldType  # must be set on each subclass
    output_description: str = "string"  # describes the Python type of the output value

    @abstractmethod
    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        """Convert a raw LLM string to a canonical Python value."""

    def __repr__(self) -> str:
        return f"<DataType:{self.field_type.value}>"


# ─── Text ─────────────────────────────────────────────────────────────────────


class TextType(DataType):
    field_type = FieldType.TEXT
    output_description = "str — trimmed whitespace"

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        return raw.strip(), None


# ─── Currency ─────────────────────────────────────────────────────────────────


class CurrencyType(DataType):
    """
    Output: {"amount": float, "currency": str, "formatted": str}
    Example: {"amount": 1234.56, "currency": "USD", "formatted": "USD 1,234.56"}
    """

    field_type = FieldType.CURRENCY
    output_description = 'dict — {"amount": float, "currency": str, "formatted": str}'

    _SYMBOLS: dict[str, str] = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₹": "INR",
    }
    _ISO_RE = re.compile(r"\b([A-Z]{3})\b")

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        target = field_def.currency_code or "USD"
        warning: str | None = None

        # Detect currency before aggressive space removal
        detected = target
        for sym, code in self._SYMBOLS.items():
            if sym in raw:
                detected = code
                break
        else:
            m = self._ISO_RE.search(raw)
            if m:
                detected = m.group(1)

        cleaned = raw.replace(",", "").replace(" ", "")

        # Remove the detected symbols/codes from the raw amount string
        for sym in self._SYMBOLS:
            cleaned = cleaned.replace(sym, "")
        m_iso = self._ISO_RE.search(raw)
        if m_iso:
            cleaned = cleaned.replace(m_iso.group(1), "")

        m = re.search(r"-?\d+(\.\d+)?", cleaned)
        if not m:
            log.debug("CurrencyType: could not parse %r", raw)
            return raw, f"Could not parse currency value from: {raw!r}"

        amount = float(m.group(0))
        if detected != target:
            warning = (
                f"Value appears to be in {detected}; "
                f"conversion to {target} not performed — original currency preserved."
            )
        return {
            "amount": amount,
            "currency": detected,
            "formatted": f"{detected} {amount:,.2f}",
        }, warning


# ─── Date ─────────────────────────────────────────────────────────────────────


class DateType(DataType):
    """Output: ISO 8601 date string, e.g. "2024-01-15"."""

    field_type = FieldType.DATE
    output_description = "str — ISO 8601 date (YYYY-MM-DD)"

    _FORMATS = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%B %Y",
        "%b %Y",
        "%Y",
    ]

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        cleaned = raw.strip().rstrip(".")
        for fmt in self._FORMATS:
            try:
                return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d"), None
            except ValueError:
                continue
        m = re.search(r"\b(19|20)\d{2}\b", cleaned)
        if m:
            return m.group(0), f"Only year could be parsed from: {raw!r}"
        log.debug("DateType: could not parse %r", raw)
        return raw, f"Could not parse date from: {raw!r}"


# ─── Category ────────────────────────────────────────────────────────────────


class CategoryType(DataType):
    """Maps the raw value to one of the allowed categories via exact then fuzzy match."""

    field_type = FieldType.CATEGORY
    output_description = "str — one of the allowed category values"

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        allowed = field_def.categories
        if not allowed:
            return raw, None

        lower_map = {c.lower(): c for c in allowed}

        if raw.lower() in lower_map:
            return lower_map[raw.lower()], None

        matches = get_close_matches(raw.lower(), lower_map.keys(), n=1, cutoff=0.6)
        if matches:
            canonical = lower_map[matches[0]]
            log.debug("CategoryType: fuzzy-matched %r -> %r", raw, canonical)
            return canonical, f"Fuzzy-matched {raw!r} → {canonical!r}"

        return raw, f"Value {raw!r} not in allowed categories: {allowed}"


# ─── Number ──────────────────────────────────────────────────────────────────


class NumberType(DataType):
    field_type = FieldType.NUMBER
    output_description = "float"

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        cleaned = raw.replace(",", "").replace(" ", "")
        m = re.search(r"-?\d+(\.\d+)?", cleaned)
        if m:
            return float(m.group(0)), None
        return raw, f"Could not parse number from: {raw!r}"


# ─── Boolean ─────────────────────────────────────────────────────────────────


class BooleanType(DataType):
    field_type = FieldType.BOOLEAN
    output_description = "bool"

    _TRUE = {"true", "yes", "y", "1", "on", "correct", "affirmative", "✓", "x"}
    _FALSE = {"false", "no", "n", "0", "off", "incorrect", "negative", "✗"}

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        lower = raw.lower().strip()
        if lower in self._TRUE:
            return True, None
        if lower in self._FALSE:
            return False, None
        return raw, f"Could not parse boolean from: {raw!r}"


# ─── Percentage ───────────────────────────────────────────────────────────────


class PercentageType(DataType):
    """Output: float — e.g. 25.5 for "25.5%"."""

    field_type = FieldType.PERCENTAGE
    output_description = "float — numeric percentage value without the % sign"

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        cleaned = raw.replace("%", "").replace(" ", "").replace(",", "")
        m = re.search(r"-?\d+(\.\d+)?", cleaned)
        if m:
            return float(m.group(0)), None
        return raw, f"Could not parse percentage from: {raw!r}"


# ─── Phone ───────────────────────────────────────────────────────────────────


class PhoneType(DataType):
    """Output: E.164-style string, e.g. "+15551234567"."""

    field_type = FieldType.PHONE
    output_description = "str — E.164 phone number (+<country_code><number>)"

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        digits = re.sub(r"\D", "", raw)
        if not digits:
            return raw, f"No digits found in phone: {raw!r}"
        if len(digits) == 10:
            digits = "1" + digits
        return "+" + digits, None


# ─── Email ───────────────────────────────────────────────────────────────────


class EmailType(DataType):
    """Output: lowercase email address."""

    field_type = FieldType.EMAIL
    output_description = "str — lowercase RFC 5322 email address"

    _RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

    def standardize(self, raw: str, field_def: FieldDefinition) -> StdResult:
        m = self._RE.search(raw)
        if m:
            return m.group(0).lower(), None
        return raw, f"Could not parse email from: {raw!r}"


# ─── Registry & dispatcher ────────────────────────────────────────────────────

_REGISTRY: dict[FieldType, DataType] = {}


def _register(cls: type[DataType]) -> type[DataType]:
    instance = cls()
    _REGISTRY[instance.field_type] = instance
    return cls


for _cls in [
    TextType,
    CurrencyType,
    DateType,
    CategoryType,
    NumberType,
    BooleanType,
    PercentageType,
    PhoneType,
    EmailType,
]:
    _register(_cls)


def standardize(raw: str, field_def: FieldDefinition) -> StdResult:
    """
    Public entry point. Looks up the correct DataType subclass from the
    registry and delegates to its standardize() method.
    """
    if not raw or raw.lower() in ("null", "none", "n/a", "not found", ""):
        warning = "Required field not found." if field_def.required else None
        return None, warning

    handler = _REGISTRY.get(field_def.type)
    if handler is None:
        log.warning(
            "No DataType registered for %s — falling back to text", field_def.type
        )
        handler = _REGISTRY[FieldType.TEXT]

    return handler.standardize(raw, field_def)


def get_type_info() -> dict[str, str]:
    """Return {type_name: output_description} for all registered types."""
    return {ft.value: inst.output_description for ft, inst in _REGISTRY.items()}
