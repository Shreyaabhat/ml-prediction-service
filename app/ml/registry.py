"""Locations of versioned model artifacts."""
import re
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_VERSION_PATTERN = re.compile(r"v\d+")


def model_dir(version: str) -> Path:
    """Return models/<version>/, rejecting anything that isn't like 'v1', 'v2', ...

    Validating the version matters later: it will come from config or a request,
    and a value like '../../etc' must never become a file path (path traversal).
    """
    if not _VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Invalid model version {version!r}; expected e.g. 'v1'")
    return MODELS_DIR / version