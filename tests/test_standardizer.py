import pytest

from models import FieldDefinition, FieldType
from processing.standardize import standardize


def _field(type: FieldType, **kwargs) -> FieldDefinition:
    return FieldDefinition(name="test", key="test", type=type, **kwargs)


class TestStandardizer:
    # --- Text ---
    def test_text_standardization(self):
        fd = _field(FieldType.TEXT)
        assert standardize("  hello world  ", fd) == ("hello world", None)
        assert standardize("\nfoo\t", fd) == ("foo", None)

    # --- Number ---
    def test_number_standardization(self):
        fd = _field(FieldType.NUMBER)
        assert standardize("123", fd) == (123.0, None)
        assert standardize("1,234.56", fd) == (1234.56, None)
        assert standardize("-42.5", fd) == (-42.5, None)
        assert standardize("not a number", fd) == (
            "not a number",
            "Could not parse number from: 'not a number'",
        )

    # --- Boolean ---
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("true", True),
            ("YES", True),
            ("y", True),
            ("1", True),
            ("on", True),
            ("correct", True),
            ("✓", True),
            ("x", True),
            ("false", False),
            ("NO", False),
            ("n", False),
            ("0", False),
            ("off", False),
            ("incorrect", False),
            ("negative", False),
            ("✗", False),
        ],
    )
    def test_boolean_standardization(self, raw, expected):
        fd = _field(FieldType.BOOLEAN)
        assert standardize(raw, fd) == (expected, None)

    def test_boolean_failure(self):
        fd = _field(FieldType.BOOLEAN)
        assert standardize("maybe", fd) == (
            "maybe",
            "Could not parse boolean from: 'maybe'",
        )

    # --- Currency ---
    def test_currency_standardization(self):
        fd = _field(FieldType.CURRENCY, currency_code="USD")

        # Exact match
        val, warn = standardize("$1,234.56", fd)
        assert val["amount"] == 1234.56
        assert val["currency"] == "USD"
        assert warn is None

        # Mismatched currency detection
        val, warn = standardize("€ 100", fd)
        assert val["amount"] == 100.0
        assert val["currency"] == "EUR"
        assert "appears to be in EUR" in warn

        # ISO Code
        val, warn = standardize("GBP 50.00", fd)
        assert val["currency"] == "GBP"
        assert val["amount"] == 50.0

    # --- Date ---
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2024-01-15", "2024-01-15"),
            ("15/01/2024", "2024-01-15"),
            ("01/15/2024", "2024-01-15"),
            ("15.01.2024", "2024-01-15"),
            ("January 15, 2024", "2024-01-15"),
            ("15 Jan 2024", "2024-01-15"),
        ],
    )
    def test_date_standardization(self, raw, expected):
        fd = _field(FieldType.DATE)
        assert standardize(raw, fd) == (expected, None)

    def test_date_year_fallback(self):
        fd = _field(FieldType.DATE)
        assert standardize("The year was 1999", fd) == (
            "1999",
            "Only year could be parsed from: 'The year was 1999'",
        )

    # --- Category ---
    def test_category_standardization(self):
        fd = _field(FieldType.CATEGORY, categories=["Apple", "Banana", "Cherry"])

        # Exact
        assert standardize("Apple", fd) == ("Apple", None)
        # Case insensitive
        assert standardize("banana", fd) == ("Banana", None)
        # Fuzzy
        val, warn = standardize("Chery", fd)
        assert val == "Cherry"
        assert "Fuzzy-matched" in warn
        # Failure
        val, warn = standardize("Dragonfruit", fd)
        assert val == "Dragonfruit"
        assert "not in allowed categories" in warn

    # --- Percentage ---
    def test_percentage_standardization(self):
        fd = _field(FieldType.PERCENTAGE)
        assert standardize("25.5%", fd) == (25.5, None)
        assert standardize(" 10 ", fd) == (10.0, None)

    # --- Phone ---
    def test_phone_standardization(self):
        fd = _field(FieldType.PHONE)
        assert standardize("+1 (555) 123-4567", fd) == ("+15551234567", None)
        assert standardize("5551234567", fd) == ("+15551234567", None)
        assert standardize("invalid", fd) == (
            "invalid",
            "No digits found in phone: 'invalid'",
        )

    # --- Email ---
    def test_email_standardization(self):
        fd = _field(FieldType.EMAIL)
        assert standardize(" TEST@example.com ", fd) == ("test@example.com", None)
        assert standardize("no email here", fd) == (
            "no email here",
            "Could not parse email from: 'no email here'",
        )

    # --- Null Handling ---
    def test_null_handling(self):
        fd = _field(FieldType.TEXT, required=False)
        assert standardize("", fd) == (None, None)
        assert standardize("N/A", fd) == (None, None)
        assert standardize("none", fd) == (None, None)

        fd_req = _field(FieldType.TEXT, required=True)
        assert standardize("null", fd_req) == (None, "Required field not found.")
