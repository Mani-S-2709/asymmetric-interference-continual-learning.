"""Run one frozen Phase 4 method/order/seed after comparison approval."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.prepare_data import prepare_xnli
from phase2.protocol import build_manifest, load_protocol
from phase2.results import make_sequential_record, validate_result_record, write_result_record
from phase2.run_baseline import (
    _create_sequential_attempt,
    _new_trainer,
    _runtime_metadata,
    load_prepared_datasets,
    run_baseline,
    validate_prepared_artifacts,
)
from phase4.interference_matrix import (
    load_phase4_matrix,
)
from phase4.interference_replay import AsymmetricInterferenceReplaySelector


DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "phase4"
    / "configs"
    / "controlled_10lang_prior_comparison_protocol.json"
)
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT / "phase4" / "artifacts" / "controlled_10lang_prior_comparison"
)
DEFAULT_INTERFERENCE_MATRIX = (
    PROJECT_ROOT / "phase4" / "artifacts" / "interference_matrix_10lang.csv"
)
METHOD_IDS = tuple(f"B{index}" for index in range(9)) + ("M3",)


def _require_comparison_gate(
    protocol: Mapping[str, Any],
    method_id: str,
) -> None:
    phase4_config = protocol.get("phase4")
    if not isinstance(phase4_config, Mapping):
        raise ValueError("The protocol is missing its Phase 4 configuration.")
    imputation = phase4_config.get("imputation")
    if (
        not isinstance(imputation, Mapping)
        or imputation.get("method") != "uniform_mean_of_k_nearest_languages"
        or imputation.get("distance") != "URIEL+ syntactic"
        or imputation.get("k") != 5
    ):
        raise ValueError("The Phase 4 pilot freezes Arabic imputation at URIEL+ k=5.")
    m3_config = phase4_config.get("m3")
    if (
        not isinstance(m3_config, Mapping)
        or m3_config.get("replay_examples_per_current_example") != 1
        or m3_config.get("replay_update_timing") != "after_each_language_stage"
    ):
        raise ValueError("The Phase 4 pilot freezes M3 at 1:1 stagewise replay.")
    if (
        phase4_config.get("analysis_tier") != "pilot"
        or phase4_config.get("statistical_claims") != "descriptive_only"
    ):
        raise ValueError("The single-seed Phase 4 protocol must remain descriptive-only.")
    gate = phase4_config.get("prior_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("The protocol is missing its Phase 4 comparison-approval gate.")
    approved_methods = gate.get("approved_methods")
    scheduled_methods = protocol.get("run_methods")
    comparison = phase4_config.get("comparison")
    training = protocol.get("training")
    valid_approved_methods = (
        isinstance(approved_methods, list)
        and bool(approved_methods)
        and all(isinstance(method_id, str) for method_id in approved_methods)
        and len(set(approved_methods)) == len(approved_methods)
        and set(approved_methods) <= set(METHOD_IDS)
    )
    valid_scheduled_methods = (
        isinstance(scheduled_methods, list)
        and bool(scheduled_methods)
        and all(isinstance(method_id, str) for method_id in scheduled_methods)
        and len(set(scheduled_methods)) == len(scheduled_methods)
        and valid_approved_methods
        and set(scheduled_methods) <= set(approved_methods)
    )
    if (
        gate.get("status") != "passed"
        or gate.get("decision_type") != "supervisor_approved_comparison"
        or gate.get("approval_recorded") is not True
        or not isinstance(gate.get("approval_date"), str)
        or not isinstance(gate.get("approval_note"), str)
        or gate.get("selected_prior") is not None
        or not valid_approved_methods
        or not valid_scheduled_methods
        or not isinstance(comparison, Mapping)
        or comparison.get("methods") != approved_methods
        or not isinstance(comparison.get("design"), str)
        or not comparison["design"].strip()
        or comparison.get("training_seed") != 42
        or not isinstance(training, Mapping)
        or training.get("seeds") != [42]
    ):
        raise RuntimeError(
            "Phase 4 training is blocked until the supervisor-approved "
            "method comparison gate is recorded in its protocol."
        )
    if method_id not in approved_methods:
        raise ValueError(
            f"Method {method_id!r} is not included in the approved Phase 4 comparison."
        )
    if method_id not in scheduled_methods:
        raise ValueError(
            f"Method {method_id!r} is approved but not scheduled by this Phase 4 protocol."
        )


def _validate_training_seed(
    protocol: Mapping[str, Any],
    training_seed: int,
) -> None:
    seeds = protocol["training"]["seeds"]
    if isinstance(training_seed, bool) or training_seed not in seeds:
        raise ValueError(
            f"Training seed {training_seed!r} is not configured in the Phase 4 protocol."
        )


def _ensure_prepared_data(
    protocol_path: str | Path,
    artifact_root: str | Path,
) -> None:
    root = Path(artifact_root)
    if (root / "protocol_manifest.json").is_file():
        return
    if (root / "data").exists():
        raise FileExistsError(
            "A data directory exists without a protocol manifest. Inspect the "
            "incomplete artifacts before preparing data."
        )
    print("Preparing the frozen XNLI subsets...")
    prepare_xnli(protocol_path, root)


def _prepare_m3_stage_datasets(
    task_order: Sequence[str],
    train_sets: Mapping[str, Any],
    selector: AsymmetricInterferenceReplaySelector,
    *,
    capacity: int,
    replay_seed: int,
    training_seed: int,
    replay_ratio: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from datasets import Dataset
    from shared.continual_learning_lora import ReplayBuffer

    replay_buffer = ReplayBuffer(capacity, seed=replay_seed)
    stage_train_sets = {}
    stage_metadata = []
    for stage_index, language in enumerate(task_order):
        current_examples = [dict(example) for example in train_sets[language]]
        if not current_examples:
            raise ValueError(f"Training data for {language!r} cannot be empty.")

        candidates = replay_buffer.snapshot()
        replay_examples = (
            selector.select_with_replacement(
                candidates,
                len(current_examples) * replay_ratio,
                language,
            )
            if candidates
            else []
        )
        replay_language_counts: dict[str, int] = {}
        for item in replay_examples:
            if item.language not in task_order[:stage_index]:
                raise RuntimeError(
                    f"M3 replay sampled unseen language {item.language!r} "
                    f"before stage {language!r}."
                )
            replay_language_counts[item.language] = (
                replay_language_counts.get(item.language, 0) + 1
            )

        combined_examples = current_examples + [
            dict(item.example) for item in replay_examples
        ]
        stage_train_sets[language] = Dataset.from_list(combined_examples).shuffle(
            seed=training_seed + stage_index
        )
        replay_buffer.extend(language, current_examples)
        stage_metadata.append({
            "stage_index": stage_index,
            "language": language,
            "current_examples": len(current_examples),
            "replay_examples": len(replay_examples),
            "training_examples": len(combined_examples),
            "replay_examples_by_language": replay_language_counts,
            "buffer_size_after_stage": len(replay_buffer),
        })
    return stage_train_sets, stage_metadata


def run_m3(
    *,
    protocol_path: str | Path,
    artifact_root: str | Path,
    order_id: str,
    training_seed: int,
    interference_matrix_path: str | Path = DEFAULT_INTERFERENCE_MATRIX,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> dict[str, Any]:
    import torch
    from huggingface_hub import get_token
    from transformers import set_seed

    from shared.continual_learning_lora import run_sequential_experiment

    protocol = load_protocol(protocol_path)
    _require_comparison_gate(protocol, "M3")
    _validate_training_seed(protocol, training_seed)
    prepared_manifest = validate_prepared_artifacts(protocol, artifact_root)
    run_manifest = build_manifest(protocol)
    run_manifest["data_artifacts"] = prepared_manifest["data_artifacts"]
    run_manifest["data_manifest_sha256"] = prepared_manifest["data_manifest_sha256"]
    run = next(
        (
            item for item in run_manifest["runs"]
            if item["method_id"] == "M3"
            and item["order_id"] == order_id
            and item["training_seed"] == training_seed
        ),
        None,
    )
    if run is None:
        raise ValueError(
            f"No M3 run is scheduled for order {order_id!r}, seed {training_seed}."
        )

    results_dir = Path(artifact_root) / "results"
    result_path = results_dir / f"{run['run_id']}.json"
    if result_path.exists():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result_record(record, run_manifest)
        print(f"Already completed; skipping {run['run_id']}.")
        return record

    scores = load_phase4_matrix(interference_matrix_path)
    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is required. Add HF_TOKEN as a Kaggle secret."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU runtime.")

    checkpoint_dir, ledger_dir = _create_sequential_attempt(
        Path(artifact_root),
        run["run_id"],
    )
    replay_config = protocol["replay"]
    phase4_config = protocol["phase4"]
    m3_config = phase4_config["m3"]
    replay_seed = replay_config["seed"]
    replay_ratio = m3_config["replay_examples_per_current_example"]
    set_seed(training_seed)
    train_sets, test_sets = load_prepared_datasets(protocol, artifact_root)
    selector = AsymmetricInterferenceReplaySelector(scores, seed=replay_seed)
    stage_train_sets, replay_stages = _prepare_m3_stage_datasets(
        run["task_order"],
        train_sets,
        selector,
        capacity=replay_config["buffer_capacity_examples"],
        replay_seed=replay_seed,
        training_seed=training_seed,
        replay_ratio=replay_ratio,
    )
    trainer = _new_trainer(protocol, training_seed, checkpoint_dir, token)
    trainer.load_model_and_tokenizer()
    matrix_metadata_path = Path(interference_matrix_path).with_name(
        f"{Path(interference_matrix_path).stem}_metadata.json"
    )
    matrix_metadata = json.loads(matrix_metadata_path.read_text(encoding="utf-8"))
    run_metadata = {
        **_runtime_metadata(protocol, "M3"),
        "protocol_sha256": run_manifest["protocol_sha256"],
        "order_id": order_id,
        "task_order": list(run["task_order"]),
        "training_seed": training_seed,
        "replay_seed": replay_seed,
        "replay_policy": "asymmetric_published_interference_matrix",
        "replay_buffer_capacity_examples": replay_config[
            "buffer_capacity_examples"
        ],
        "replay_examples_per_current_example": replay_ratio,
        "replay_update_timing": m3_config["replay_update_timing"],
        "replay_stages": replay_stages,
        "interference_matrix_sha256": matrix_metadata["matrix_sha256"],
        "interference_source_matrix_sha256": matrix_metadata[
            "source_matrix_sha256"
        ],
        "interference_imputation": matrix_metadata["imputation"],
        "checkpoint_attempt": checkpoint_dir.name,
        "method_deviation": (
            "Arabic row and column scores are uniform means over the five "
            "nearest covered languages in URIEL+ syntactic-distance space. "
            "M3 samples exactly one replay example per current example with "
            "replacement; the first language stage has no replay."
        ),
    }
    kwargs: dict[str, Any] = {}
    if wandb_project:
        kwargs["wandb_project"] = wandb_project
        kwargs["wandb_entity"] = wandb_entity
    result = run_sequential_experiment(
        trainer,
        task_order=run["task_order"],
        train_datasets=stage_train_sets,
        test_datasets=test_sets,
        experiment_id=run["run_id"],
        output_dir=ledger_dir,
        dataset_name=protocol["dataset"]["id"],
        task_name="xnli",
        run_metadata=run_metadata,
        **kwargs,
    )
    matrix = result["ledger"].performance_matrix()
    zero_shot = result["ledger"].baseline_accuracies()
    if any(value is None for row in matrix for value in row) or any(
        value is None for value in zero_shot
    ):
        raise RuntimeError("M3 sequential experiment returned incomplete evaluations.")
    record = make_sequential_record(
        manifest=run_manifest,
        run=run,
        performance_matrix=matrix,
        zero_shot_accuracy=zero_shot,
        metadata=run_metadata,
    )
    write_result_record(record, results_dir, run_manifest)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def run_selected_pair(
    *,
    method_id: str,
    order_id: str,
    training_seed: int,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    interference_matrix_path: str | Path = DEFAULT_INTERFERENCE_MATRIX,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    if method_id not in METHOD_IDS:
        raise ValueError(f"method_id must be one of {METHOD_IDS}.")
    protocol = load_protocol(protocol_path)
    _require_comparison_gate(protocol, method_id)
    _validate_training_seed(protocol, training_seed)
    order_ids = {order["id"] for order in protocol["orders"]}
    if order_id not in order_ids:
        raise ValueError(f"Unknown Phase 4 order ID: {order_id!r}.")

    if method_id == "M3":
        load_phase4_matrix(interference_matrix_path)
        _ensure_prepared_data(protocol_path, artifact_root)
        return run_m3(
            protocol_path=protocol_path,
            artifact_root=artifact_root,
            order_id=order_id,
            training_seed=training_seed,
            interference_matrix_path=interference_matrix_path,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )

    _ensure_prepared_data(protocol_path, artifact_root)
    baseline_order_id = (
        "order-independent" if method_id == "B1" else order_id
    )
    return run_baseline(
        method_id=method_id,
        protocol_path=protocol_path,
        artifact_root=artifact_root,
        order_ids=[baseline_order_id],
        training_seeds=[training_seed],
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_IDS, required=True)
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--interference-matrix",
        type=Path,
        default=DEFAULT_INTERFERENCE_MATRIX,
        help=(
            "Prepared ten-language matrix CSV; required for M3 and created by "
            "phase4.prepare_interference_matrix."
        ),
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()

    result = run_selected_pair(
        method_id=args.method,
        order_id=args.order_id,
        training_seed=args.seed,
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        interference_matrix_path=args.interference_matrix,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    records = result if isinstance(result, list) else [result]
    for record in records:
        print(
            f"Completed {record['run_id']}: "
            f"ACC={record['metrics']['ACC']:.4f}"
        )


if __name__ == "__main__":
    main()
