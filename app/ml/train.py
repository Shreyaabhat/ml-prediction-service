"""Train the intent classifier and save a versioned artifact.

Run from the project root (venv active):
    python -m app.ml.train --version v1

Use `-m` (module mode), NOT `python app/ml/train.py`. joblib saves functions by
reference; running as a module keeps normalize_text's path as
'app.ml.preprocess.normalize_text', which the API can import later.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from app.ml.evaluate import find_best_threshold, threshold_metrics, top_prediction
from app.ml.preprocess import load_dataset, normalize_text
from app.ml.registry import model_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "clinc150_data_full.json"

HYPERPARAMETERS = {
    "tfidf_ngram_range": [1, 2],
    "tfidf_sublinear_tf": True,
    "logreg_C": 10.0,
    "logreg_max_iter": 1000,
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=tuple(HYPERPARAMETERS["tfidf_ngram_range"]),
                    sublinear_tf=HYPERPARAMETERS["tfidf_sublinear_tf"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=HYPERPARAMETERS["logreg_C"],
                    max_iter=HYPERPARAMETERS["logreg_max_iter"],
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the intent classifier.")
    parser.add_argument("--version", default="v1", help="Model version label, e.g. v1")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        sys.exit("Dataset not found. Run: python scripts/download_data.py")

    out_dir = model_dir(args.version)
    if (out_dir / "model.joblib").exists():
        sys.exit(f"{args.version} already exists. Artifacts are immutable; choose a new version.")

    data = load_dataset(DATA_PATH)
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    x_test, y_test = data["test"]

    print(f"Training on {len(x_train)} examples...")
    model = build_pipeline()
    model.fit(x_train, y_train)

    # 1) Choose the threshold on VALIDATION data only.
    val_pred, val_conf = top_prediction(model, x_val)
    _, oos_val_conf = top_prediction(model, data["oos_val"])
    best = find_best_threshold(y_val, val_pred, val_conf, oos_val_conf)
    threshold = best["threshold"]

    # 2) Evaluate ONCE on TEST data with the frozen threshold.
    test_pred, test_conf = top_prediction(model, x_test)
    _, oos_test_conf = top_prediction(model, data["oos_test"])
    test_metrics = threshold_metrics(y_test, test_pred, test_conf, oos_test_conf, threshold)
    test_metrics["raw_accuracy_no_threshold"] = float(accuracy_score(y_test, test_pred))
    test_metrics["macro_f1_no_threshold"] = float(f1_score(y_test, test_pred, average="macro"))

    metadata = {
        "model_version": args.version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "CLINC150 (data_full.json)",
            "citation": "Larson et al., 2019, EMNLP-IJCNLP",
            "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "sizes": {
                "train": len(x_train),
                "val": len(x_val),
                "test": len(x_test),
                "oos_val": len(data["oos_val"]),
                "oos_test": len(data["oos_test"]),
            },
        },
        "num_intents": int(len(model.classes_)),
        "hyperparameters": HYPERPARAMETERS,
        "confidence_threshold": threshold,
        "threshold_selection": {
            "method": "maximize mean(in_scope_accuracy, oos_recall) on validation data",
            "validation_score": best["score"],
            "validation_metrics": best["metrics"],
        },
        "test_metrics": test_metrics,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    out_dir.mkdir(parents=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved artifact to {out_dir}")
    print(f"Chosen confidence threshold: {threshold}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()