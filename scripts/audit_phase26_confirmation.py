"""Read-only case-level audit of the completed Phase 2.6 sealed confirmation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.form_variation import (
    _device_from_name,
    _load_phase25_checkpoint,
    build_phase26_sealed_pressure_test,
    operation_confusion_matrix,
    predict_operation_labels_batched,
)


def main() -> None:
    config_path = Path("configs/phase26_confirmation.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["phase26_confirmation"]
    device = _device_from_name(str(config["training"].get("device", "auto")))
    sealed = build_phase26_sealed_pressure_test(tuple(tuple(pair) for pair in experiment["sealed_pressure_pairs"]))
    output: dict[str, object] = {"source": str(config_path), "variants": {}}
    root = Path(experiment["output_dir"])
    for variants in experiment["variants_per_operation"]:
        cell: dict[str, object] = {}
        for arm in ("baseline", "minimal_lexical_contrast"):
            arm_audit: dict[str, object] = {}
            for seed in experiment["seeds"]:
                checkpoint = root / f"variants_{variants}" / arm / f"seed_{seed}"
                model, tokenizer = _load_phase25_checkpoint(checkpoint, config, device)
                tracks: dict[str, object] = {}
                for track, examples in sealed.items():
                    predictions = predict_operation_labels_batched(model, tokenizer, examples, device, batch_size=64)
                    tracks[track] = {
                        "accuracy": sum(pred == example.target for example, pred in zip(examples, predictions)) / len(examples),
                        "confusion_matrix": operation_confusion_matrix(examples, predictions),
                        "errors": [
                            {"utterance": example.utterance, "target": example.target, "prediction": prediction}
                            for example, prediction in zip(examples, predictions)
                            if prediction != example.target
                        ],
                    }
                arm_audit[str(seed)] = tracks
            cell[arm] = arm_audit
        output["variants"][str(variants)] = cell
    (root / "failure_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
