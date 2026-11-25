"""Run a configurable no-replay NLI stream with per-language data sources."""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE1_SOURCE = PROJECT_ROOT / "phase1" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PHASE1_SOURCE) not in sys.path:
    sys.path.insert(0, str(PHASE1_SOURCE))

from continual_learning.nli_data import (
    prepare_nli_stream,
)
from continual_learning.stream_loader import (
    HuggingFaceStreamLoader,
    StreamStage,
)
from shared.continual_learning_lora import (
    LlamaLoRATrainer,
    evaluate_xnli_accuracy,
    run_sequential_experiment,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "phase1" / "configs" / "en_hi_bn_indicxnli.json"
)


def _read_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Experiment config must be a JSON object.")
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported experiment config schema_version.")
    return value


def _validate_config(config: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        experiment_id,
    ):
        raise ValueError("experiment_id must be a safe path component.")

    stream = config.get("stream")
    if not isinstance(stream, list) or len(stream) < 2:
        raise ValueError("stream must contain at least two language stages.")
    languages = []
    for item in stream:
        if not isinstance(item, dict):
            raise ValueError("Each stream entry must be an object.")
        language = item.get("language")
        dataset_id = item.get("dataset_id")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Each stream entry needs a language code.")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("Each stream entry needs a dataset_id.")
        languages.append(language)
    if len(set(languages)) != len(languages):
        raise ValueError("Each language may appear only once in the stream.")

    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be a non-empty list of unique integers.")

    selection = config.get("data_selection")
    if not isinstance(selection, dict):
        raise ValueError("data_selection must be an object.")
    for key in ("train_examples_per_language", "test_examples_per_language"):
        value = selection.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    test_seed = selection.get("test_selection_seed")
    if isinstance(test_seed, bool) or not isinstance(test_seed, int):
        raise ValueError("test_selection_seed must be an integer.")

    model = config.get("model")
    training = config.get("training")
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise ValueError("model.id must be configured.")
    if not isinstance(training, dict):
        raise ValueError("training must be an object.")
    if not isinstance(config.get("require_parallel_label_alignment", False), bool):
        raise ValueError("require_parallel_label_alignment must be boolean.")
    return languages, seeds


def _make_stages(config: Mapping[str, Any]) -> list[StreamStage]:
    task = config.get("task", "nli")
    return [
        StreamStage(
            language=item["language"],
            task=task,
            split="train",
            dataset_id=item["dataset_id"],
            dataset_config=item.get("dataset_config"),
            revision=item.get("revision"),
        )
        for item in config["stream"]
    ]


def prepare_nli_datasets(
    config: Mapping[str, Any],
    *,
    dataset_loader: Any | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load mixed-source NLI data and freeze reproducible sample selections."""
    languages, seeds = _validate_config(config)
    selection = config["data_selection"]
    split_names = config.get("splits", {"train": "train", "test": "test"})
    if not isinstance(split_names, dict):
        raise ValueError("splits must be an object.")
    train_split = split_names.get("train", "train")
    test_split = split_names.get("test", "test")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (train_split, test_split)
    ):
        raise ValueError("splits.train and splits.test must be non-empty strings.")
    stages = _make_stages(config)
    loader = HuggingFaceStreamLoader(
        stages,
        shuffle_train=False,
        dataset_loader=dataset_loader,
    )
    train_by_seed, test_by_language, data_manifest = prepare_nli_stream(
        stages,
        loader,
        experiment_id=config["experiment_id"],
        seeds=seeds,
        train_count=selection["train_examples_per_language"],
        test_count=selection["test_examples_per_language"],
        test_seed=selection["test_selection_seed"],
        train_split=train_split,
        test_split=test_split,
        require_parallel_label_alignment=config.get(
            "require_parallel_label_alignment",
            False,
        ),
    )
    for language in languages:
        source = data_manifest["sources"][language]
        print(
            f"{language}: {source['dataset_id']}[{source['dataset_config']}] "
            f"train={selection['train_examples_per_language']}, "
            f"test={selection['test_examples_per_language']}"
        )
    return train_by_seed, test_by_language, data_manifest


def _make_trainer(
    config: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> LlamaLoRATrainer:
    model = config["model"]
    training = config["training"]
    return LlamaLoRATrainer(
        model_name=model["id"],
        output_dir=output_dir,
        task_mode=model.get("task_mode", "causal_lm"),
        max_length=training["max_length"],
        learning_rate=training["learning_rate"],
        train_batch_size=training["train_batch_size"],
        eval_batch_size=training.get("eval_batch_size", 1),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        num_train_epochs=training["num_train_epochs"],
        seed=seed,
        lora_rank=training["lora_rank"],
        lora_alpha=training["lora_alpha"],
        lora_dropout=training["lora_dropout"],
        load_in_4bit=training["load_in_4bit"],
        gradient_checkpointing=training["gradient_checkpointing"],
        token=True,
    )


def _make_report(
    config: Mapping[str, Any],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    languages, seeds = _validate_config(config)
    matrices = [result["matrix"] for result in seed_results]
    mean_matrix = [
        [
            statistics.mean(matrix[row][column] for matrix in matrices)
            for column in range(len(languages))
        ]
        for row in range(len(languages))
    ]
    std_matrix = [
        [
            statistics.stdev(matrix[row][column] for matrix in matrices)
            if len(matrices) > 1
            else 0.0
            for column in range(len(languages))
        ]
        for row in range(len(languages))
    ]
    seed_checks = []
    for result in seed_results:
        matrix = result["matrix"]
        diagonal = [matrix[index][index] for index in range(len(languages))]
        previous_final_mean = statistics.mean(matrix[-1][:-1])
        previous_learned_mean = statistics.mean(diagonal[:-1])
        last_language_strong = matrix[-1][-1] >= previous_final_mean
        earlier_languages_degraded = previous_final_mean < previous_learned_mean
        seed_checks.append(
            {
                "seed": result["seed"],
                "last_seen_language_strong_relative_to_previous_mean": (
                    last_language_strong
                ),
                "earlier_languages_degraded_on_average": (
                    earlier_languages_degraded
                ),
                "checkpoint_pattern_passed": (
                    last_language_strong and earlier_languages_degraded
                ),
            }
        )

    first_seen_mean = [
        statistics.mean(matrix[index][index] for matrix in matrices)
        for index in range(len(languages))
    ]
    return {
        "experiment_id": config["experiment_id"],
        "task_order": languages,
        "seeds": seeds,
        "train_examples_per_language": (
            config["data_selection"]["train_examples_per_language"]
        ),
        "test_examples_per_language": (
            config["data_selection"]["test_examples_per_language"]
        ),
        "mean_performance_matrix_rows_after_each_stage": mean_matrix,
        "std_performance_matrix_rows_after_each_stage": std_matrix,
        "mean_final_accuracies_by_language": dict(
            zip(languages, mean_matrix[-1])
        ),
        "mean_first_seen_accuracies_by_language": dict(
            zip(languages, first_seen_mean)
        ),
        "per_seed_results": seed_results,
        "per_seed_checkpoint_checks": seed_checks,
        "checkpoint_pattern_pass_rate": (
            sum(check["checkpoint_pattern_passed"] for check in seed_checks)
            / len(seed_checks)
        ),
        "checkpoint_pattern_passed_all_seeds": all(
            check["checkpoint_pattern_passed"] for check in seed_checks
        ),
        "interpretation_note": (
            "The checkpoint rule is unchanged. Three seeds are a pilot, "
            "not a definitive statistical conclusion."
        ),
    }


def run(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    output_root: str | Path | None = None,
) -> Path | None:
    config = _read_config(config_path)
    languages, seeds = _validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory (GiB):",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    )

    train_by_seed, test_by_language, data_manifest = prepare_nli_datasets(config)
    if preflight_only:
        trainer = _make_trainer(
            config,
            seeds[0],
            PROJECT_ROOT / "phase1" / "checkpoints" / "_preflight",
        )
        trainer.load_model_and_tokenizer()
        for language in languages:
            smoke_test = test_by_language[language].select(
                range(min(8, len(test_by_language[language])))
            )
            accuracy = evaluate_xnli_accuracy(
                trainer.model,
                trainer.tokenizer,
                smoke_test,
                language,
            )
            print(f"{language}: evaluator preflight accuracy={accuracy:.3f}")
        torch.cuda.synchronize()
        print("Dataset, model, and evaluator preflight passed.")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group_id = f"{config['experiment_id']}-{timestamp}"
    results_root = Path(output_root) if output_root else PROJECT_ROOT / "phase1" / "results"
    checkpoints_root = PROJECT_ROOT / "phase1" / "checkpoints"
    results_group_dir = results_root / group_id
    checkpoints_group_dir = checkpoints_root / group_id
    results_group_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_group_dir.mkdir(parents=True, exist_ok=False)
    (results_group_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_group_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    seed_results = []
    for seed in seeds:
        set_seed(seed)
        seed_name = f"seed-{seed}"
        result_dir = results_group_dir / seed_name / "evaluation"
        trainer = _make_trainer(
            config,
            seed,
            checkpoints_group_dir / seed_name,
        )
        trainer.load_model_and_tokenizer()
        trainer.model.print_trainable_parameters()
        result = run_sequential_experiment(
            trainer,
            task_order=languages,
            train_datasets=train_by_seed[seed],
            test_datasets=test_by_language,
            experiment_id=f"{group_id}-{seed_name}",
            output_dir=result_dir,
            dataset_name="multi_source_xnli",
            task_name=config.get("task", "nli"),
            run_metadata={
                "phase": 1,
                "method": "sequential_finetuning_no_replay",
                "seed": seed,
                "seed_group": group_id,
                "data_manifest_sha256": data_manifest["data_manifest_sha256"],
                "model": config["model"]["id"],
                "train_samples_per_language": (
                    config["data_selection"]["train_examples_per_language"]
                ),
                "test_samples_per_language": (
                    config["data_selection"]["test_examples_per_language"]
                ),
                "prompt_protocol": config["model"].get("prompt_protocol"),
            },
        )
        seed_results.append(
            {
                "seed": seed,
                "experiment_id": result["experiment_id"],
                "matrix": result["ledger"].performance_matrix(),
                "metrics": result["metrics"],
                "evaluation_json": str(result["ledger"].json_path),
            }
        )
        print(f"Completed {seed_name}: {result['ledger'].json_path}")
        del result, trainer
        gc.collect()
        torch.cuda.empty_cache()

    report = _make_report(config, seed_results)
    report["data_manifest_sha256"] = data_manifest["data_manifest_sha256"]
    report_path = results_group_dir / "multiseed_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Summary saved to: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(
        args.config,
        preflight_only=args.preflight_only,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
