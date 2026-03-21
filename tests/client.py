"""
client.py — Client script to test the Parsea Document Extraction API.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# ── Ensure src is on the path when running from backend/ ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from models import ExtractionInstructions, FieldDefinition, FieldType, LLMProvider
from utils import configure_logging, get_logger

configure_logging(level="INFO")
log = get_logger(__name__)

PDF_PATH = Path(__file__).resolve().parent.parent / "docs" / "bau.pdf"
API_URL = "http://localhost:8000/api/v1/extract"
API_TOKEN = os.environ.get("API_TOKEN", "dev-secret-token")

instructions = ExtractionInstructions(
    document_description=(
        "German building energy certificate (Energieausweis) for a residential building. "
        "Contains building details, energy performance metrics, and CO₂ emissions data."
    ),
    llm_provider=LLMProvider.HUGGINGFACE,
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    fields=[
        FieldDefinition(
            name="Building Type",
            key="building_type",
            description="Type of the building (e.g. freistehendes Mehrfamilienhaus)",
            type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="Address",
            key="address",
            description="Full street address of the building",
            type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="Number of Apartments",
            key="num_apartments",
            description="Number of residential units in the building (Anzahl der Wohnungen)",
            type=FieldType.NUMBER,
            required=True,
        ),
        FieldDefinition(
            name="Primary Energy Source (Heating)",
            key="primary_energy_source_heating",
            description="Main energy source used for heating (e.g. Fernwärme regenerativ)",
            type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="Certificate Valid Until",
            key="valid_until",
            description="Expiration date of the energy certificate (Gültig bis)",
            type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="Final Energy Demand",
            key="final_energy_demand_kwh",
            description="Endenergiebedarf in kWh/(m²·a)",
            type=FieldType.NUMBER,
            required=True,
        ),
        FieldDefinition(
            name="Primary Energy Demand",
            key="primary_energy_demand_kwh",
            description="Primärenergiebedarf in kWh/(m²·a)",
            type=FieldType.NUMBER,
            required=True,
        ),
        FieldDefinition(
            name="CO₂ Emissions",
            key="co2_emissions",
            description="Treibhausgasemissionen in kg CO₂-Äquivalent/(m²·a)",
            type=FieldType.NUMBER,
            required=True,
        ),
        FieldDefinition(
            name="Energy Efficiency Class",
            key="energy_class",
            description="Energy efficiency rating label (A+, A, B, C, D, E, F, G, H)",
            type=FieldType.CATEGORY,
            categories=["A+", "A", "B", "C", "D", "E", "F", "G", "H"],
            required=True,
        ),
        FieldDefinition(
            name="Has Ventilation System",
            key="has_ventilation_system",
            description="Whether the building has a mechanical ventilation system (Lüftungsanlage)",
            type=FieldType.BOOLEAN,
        ),
        FieldDefinition(
            name="Number of Air Conditioning Systems",
            key="num_ac_systems",
            description="Number of inspectable climate/AC systems (Inspektionspflichtige Klimaanlagen)",
            type=FieldType.NUMBER,
        ),
    ],
)


def display_results(response_data: dict) -> None:
    """Print extraction results in a readable format."""
    print("\n" + "═" * 70)
    print("  📄 EXTRACTION RESULTS")
    print("═" * 70)
    print(f"  Document:     {response_data.get('document_id')}")
    print(f"  Total pages:  {response_data.get('total_pages')}")
    print(f"  Total chunks: {response_data.get('total_chunks')}")
    print("─" * 70)

    fields = response_data.get("fields", [])
    for field in fields:
        icon = "✅" if field.get("found") else "❌"
        warn = f"  ⚠️  {field.get('warning')}" if field.get("warning") else ""

        print(f"\n  {icon} {field.get('key')}")
        print(f"     Raw:          {field.get('raw_value')!r}")

        # Pretty-print dicts / booleans
        val = field.get("value")
        if isinstance(val, dict):
            print(f"     Standardized: {json.dumps(val, ensure_ascii=False)}")
        else:
            print(f"     Standardized: {val!r}")

        confidence = field.get("confidence", 0)
        print(f"     Confidence:   {confidence:.0%}")

        reference = field.get("reference")
        if reference:
            print(
                f'     Source:       page {reference.get("page")}: "{reference.get("chunk_preview")}"'
            )
        if warn:
            print(warn)

    print("\n" + "═" * 70)


def main() -> None:
    if not PDF_PATH.exists():
        log.error("Document not found: %s", PDF_PATH)
        return

    log.info("Sending document: %s", PDF_PATH.name)

    headers = {"X-API-Key": API_TOKEN}

    with open(PDF_PATH, "rb") as f:
        files = {"file": (PDF_PATH.name, f, "application/pdf")}
        data = {"instructions_json": instructions.model_dump_json()}

        log.info("Connecting to %s...", API_URL)
        try:
            response = requests.post(API_URL, headers=headers, files=files, data=data)
            response.raise_for_status()

            result = response.json()
            display_results(result)

        except requests.exceptions.HTTPError:
            log.error("API Request failed with status %d", response.status_code)
            try:
                log.error("Response details: %s", response.json())
            except Exception:
                log.error("Response details: %s", response.text)
        except requests.exceptions.RequestException as e:
            log.error("API Request failed: %s", e)


if __name__ == "__main__":
    main()
