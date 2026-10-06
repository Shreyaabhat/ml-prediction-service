"""Load a saved model in a FRESH process and run a few predictions.

Run from the project root:  python -m app.ml.predict_demo --version v1
"""
import argparse
import json

import joblib

from app.ml.evaluate import top_prediction
from app.ml.registry import model_dir

SAMPLES = [
    "what's the weather like tomorrow",
    "set a timer for ten minutes",
    "i lost my credit card what should i do",
    "what is the best pizza topping on mars",
    "flibbertigibbet purple dishwasher",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    directory = model_dir(args.version)
    model = joblib.load(directory / "model.joblib")
    threshold = json.loads((directory / "metadata.json").read_text())["confidence_threshold"]

    labels, confidences = top_prediction(model, SAMPLES)
    for text, label, conf in zip(SAMPLES, labels, confidences, strict=True):
        shown = label if conf >= threshold else "oos (below threshold)"
        print(f"{conf:.3f}  {shown:<24} <- {text!r}")


if __name__ == "__main__":
    main()