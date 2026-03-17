import argparse
import os
import sys

import requests

API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"


def test_health() -> bool:
    """Test the root health check endpoint."""
    url = f"{API_URL}/"
    print(f"Testing health endpoint at {url}...")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_extraction(api_token: str, pdf_path: str) -> bool:
    """Test the document extraction endpoint with a sample PDF."""
    url = f"{API_URL}/api/v1/extract"
    print(f"\nTesting extraction endpoint at {url}...")

    headers = {"X-API-Key": api_token}

    # A simple instruction set for testing
    instructions = {
        "document_description": "A test document",
        "llm_provider": "huggingface",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "fields": [
            {"name": "Dummy Field", "key": "dummy", "type": "text", "required": False}
        ],
    }

    import json

    data = {"instructions_json": json.dumps(instructions)}

    try:
        with open(pdf_path, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            print("Sending request... (this may take a moment)")
            response = requests.post(
                url, headers=headers, files=files, data=data, timeout=60
            )

            response.raise_for_status()
            result = response.json()
            print(
                f"✅ Extraction passed. Received {len(result.get('fields', []))} fields."
            )
            return True
    except FileNotFoundError:
        print(f"❌ Could not find test document at {pdf_path}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Extraction failed: {e}")
        if "response" in locals() and response is not None:
            try:
                print(f"Details: {response.json()}")
            except Exception:
                print(f"Details: {response.text}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the Parsea API deployment")
    parser.add_argument(
        "--check-all", action="store_true", help="Run both health and extraction tests"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="dev-secret-token",
        help="API token for extraction test",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="docs/bau.pdf",
        help="Path to PDF for extraction test",
    )

    args = parser.parse_args()

    health_ok = test_health()

    if not health_ok:
        sys.exit(1)

    if args.check_all:
        extract_ok = test_extraction(args.token, args.pdf)
        if not extract_ok:
            sys.exit(1)

    print("\n🎉 All requested tests passed successfully!")


if __name__ == "__main__":
    main()
