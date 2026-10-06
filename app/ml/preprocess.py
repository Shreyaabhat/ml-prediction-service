"""Text normalization and dataset loading.

normalize_text is shared by training AND serving (and later the cache-key logic),
which prevents training/serving skew.
"""
import json
import re
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, trim, and collapse repeated whitespace."""
    return _WHITESPACE.sub(" ", text.strip().lower())


def load_dataset(path: Path) -> dict:
    """Load CLINC150 and return the splits we use.

    In-scope splits are (texts, labels). OOS splits are just texts, since their
    label is always "oos" and we never train on them.
    We deliberately ignore `oos_train`: we train on the 150 intents only.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def with_labels(key: str) -> tuple[list[str], list[str]]:
        return [row[0] for row in raw[key]], [row[1] for row in raw[key]]

    return {
        "train": with_labels("train"),
        "val": with_labels("val"),
        "test": with_labels("test"),
        "oos_val": [row[0] for row in raw["oos_val"]],
        "oos_test": [row[0] for row in raw["oos_test"]],
    }