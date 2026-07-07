from pathlib import Path

from fewshot_determinants.evaluation.metrics import compute_classification_metrics
from fewshot_determinants.models.dummy import MajorityClassifier
from fewshot_determinants.paths import RUNS_DIR
from fewshot_determinants.training.run_metadata import RunMetadata
from fewshot_determinants.utils.io import ensure_dir, save_json


def main():
    y_train = [1, 1, 1, 0]
    X_train = ["good", "great", "nice", "bad"]
    y_test = [1, 1, 0, 0]
    X_test = ["awesome", "fine", "terrible", "poor"]

    model = MajorityClassifier().fit(y_train)
    y_pred = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)

    metadata = RunMetadata.create("local-smoke-test")
    run_dir = ensure_dir(RUNS_DIR / metadata.run_id)
    save_json({"metadata": metadata.to_dict(), "metrics": metrics}, run_dir / "summary.json")

    print(f"Run completed: {metadata.run_id}")
    print(f"Summary file: {Path(run_dir / 'summary.json').resolve()}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
