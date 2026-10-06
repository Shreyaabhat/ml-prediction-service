"""Download the CLINC150 'full' dataset into data/.

Run from the project root:  python scripts/download_data.py
"""
import hashlib
import sys
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
DEST = Path(__file__).resolve().parents[1] / "data" / "clinc150_data_full.json"
EXPECTED_KEYS = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")

    # Always set a timeout on network calls. Without one, a hung server hangs you forever.
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    data = response.json()
    missing = EXPECTED_KEYS - set(data)
    if missing:
        sys.exit(f"Unexpected file format. Missing keys: {sorted(missing)}. Found: {sorted(data)}")

    DEST.write_bytes(response.content)
    print(f"Saved to {DEST}")
    print(f"SHA-256: {hashlib.sha256(response.content).hexdigest()}")
    for key in sorted(EXPECTED_KEYS):
        print(f"  {key}: {len(data[key])} examples")


if __name__ == "__main__":
    main()