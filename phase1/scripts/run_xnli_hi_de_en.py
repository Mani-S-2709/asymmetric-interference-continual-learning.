import argparse
import gc
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.continual_learning_lora import (
    LlamaLoRATrainer,
    evaluate_xnli_accuracy,
    run_sequential_experiment,
)


MODEL_ID = "meta-llama/Llama-3.2-1B"
DATASET_ID = "facebook/xnli"
LANGUAGES = ["hi", "de", "en"]
SEEDS = [42, 43, 44]
TEST_SPLIT_SEED = 42
TRAIN_SAMPLES_PER_LANGUAGE = 1000
TEST_SAMPLES_PER_LANGUAGE = 500
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def format_xnli_row(example: dict) -> dict[str, str]:
    label_id = int(example["label"])
    if label_id not in LABEL_NAMES:
        raise ValueError(f"Unexpected XNLI label id: {label_id}")
    prompt = (
        "Task: Natural language inference\n"
        f"Premise: {example['premise']}\n"
        f"Hypothesis: {example['hypothesis']}\n"
        "Choose one label: entailment, neutral, contradiction\n"
        "Answer:"
    )
    return {"prompt": prompt, "target": " " + LABEL_NAMES[label_id]}


def make_trainer(seed: int, output_dir: Path) -> LlamaLoRATrainer:
    return LlamaLoRATrainer(
        model_name=MODEL_ID,
        output_dir=output_dir,
        task_mode="causal_lm",
        max_length=512,
        learning_rate=2e-4,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        seed=seed,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        load_in_4bit=True,
        gradient_checkpointing=True,
        token=True,
    )


def load_xnli_datasets() -> tuple[dict, dict]:
    train_sets_by_seed = {seed: {} for seed in SEEDS}
    test_sets = {}

    for language in LANGUAGES:
        train_raw = load_dataset(DATASET_ID, language, split="train")
        test_raw = load_dataset(DATASET_ID, language, split="test")
        if len(train_raw) < TRAIN_SAMPLES_PER_LANGUAGE:
            raise ValueError(f"Not enough training examples for {language}.")
        if len(test_raw) < TEST_SAMPLES_PER_LANGUAGE:
            raise ValueError(f"Not enough test examples for {language}.")

        test_sets[language] = test_raw.shuffle(seed=TEST_SPLIT_SEED).select(
            range(TEST_SAMPLES_PER_LANGUAGE)
        )
        for seed in SEEDS:
            train_subset = train_raw.shuffle(seed=seed).select(
                range(TRAIN_SAMPLES_PER_LANGUAGE)
            )
            train_sets_by_seed[seed][language] = train_subset.map(
                format_xnli_row,
                remove_columns=train_subset.column_names,
                desc=f"Formatting {language} training examples for seed {seed}",
            )
        print(
            f"{language}: train={TRAIN_SAMPLES_PER_LANGUAGE} per seed, "
            f"fixed test={len(test_sets[language])}"
        )

    for seed in SEEDS:
        if list(train_sets_by_seed[seed]) != LANGUAGES:
            raise ValueError(f"Training data order is incorrect for seed {seed}.")
        examples = train_sets_by_seed[seed][LANGUAGES[0]].select(range(3))
        if not all(set(example) == {"prompt", "target"} for example in examples):
            raise ValueError("Formatted training examples have unexpected columns.")

    return train_sets_by_seed, test_sets


def make_experiment_report(
    experiment_group_id: str,
    seed_results: list[dict],
) -> dict:
    matrices = [result["matrix"] for result in seed_results]
    mean_matrix = [
        [
            statistics.mean(matrix[row][column] for matrix in matrices)
            for column in range(len(LANGUAGES))
        ]
        for row in range(len(LANGUAGES))
    ]
    std_matrix = [
        [
            statistics.stdev(matrix[row][column] for matrix in matrices)
            for column in range(len(LANGUAGES))
        ]
        for row in range(len(LANGUAGES))
    ]

    seed_pattern_results = []
    for seed_result in seed_results:
        matrix = seed_result["matrix"]
        final_row = matrix[-1]
        diagonal = [matrix[index][index] for index in range(len(LANGUAGES))]
        previous_final_mean = statistics.mean(final_row[:-1])
        previous_learned_mean = statistics.mean(diagonal[:-1])
        last_language_strong = final_row[-1] >= previous_final_mean
        earlier_languages_degraded = previous_final_mean < previous_learned_mean
        seed_pattern_results.append(
            {
                "seed": seed_result["seed"],
                "last_seen_language_strong_relative_to_previous_mean": last_language_strong,
                "earlier_languages_degraded_on_average": earlier_languages_degraded,
                "checkpoint_pattern_passed": (
                    last_language_strong and earlier_languages_degraded
                ),
            }
        )

    first_seen_mean = [
        statistics.mean(matrix[index][index] for matrix in matrices)
        for index in range(len(LANGUAGES))
    ]
    return {
        "experiment_group_id": experiment_group_id,
        "task_order": LANGUAGES,
        "seeds": SEEDS,
        "train_examples_per_language": TRAIN_SAMPLES_PER_LANGUAGE,
        "test_examples_per_language": TEST_SAMPLES_PER_LANGUAGE,
        "fixed_test_split_seed": TEST_SPLIT_SEED,
        "mean_performance_matrix_rows_after_each_stage": mean_matrix,
        "std_performance_matrix_rows_after_each_stage": std_matrix,
        "mean_final_accuracies_by_language": dict(zip(LANGUAGES, mean_matrix[-1])),
        "mean_first_seen_accuracies_by_language": dict(zip(LANGUAGES, first_seen_mean)),
        "per_seed_results": seed_results,
        "per_seed_checkpoint_checks": seed_pattern_results,
        "checkpoint_pattern_pass_rate": (
            sum(item["checkpoint_pattern_passed"] for item in seed_pattern_results)
            / len(seed_pattern_results)
        ),
        "checkpoint_pattern_passed_all_seeds": all(
            item["checkpoint_pattern_passed"] for item in seed_pattern_results
        ),
        "interpretation_note": (
            "Three seeds provide a preliminary repeatability check, not a definitive conclusion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local XNLI stream in Hindi, German, English order."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Prepare the datasets and verify the quantized model loads on the local GPU.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory (GiB):",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    )

    train_sets_by_seed, test_sets = load_xnli_datasets()
    if args.preflight_only:
        trainer = make_trainer(
            SEEDS[0], PROJECT_ROOT / "phase1" / "checkpoints" / "_preflight"
        )
        trainer.load_model_and_tokenizer()
        trainer.model.print_trainable_parameters()
        for language in LANGUAGES:
            smoke_test = test_sets[language].select(range(8))
            accuracy = evaluate_xnli_accuracy(
                trainer.model,
                trainer.tokenizer,
                smoke_test,
                language,
            )
            print(f"{language}: evaluator preflight accuracy={accuracy:.3f}")
        torch.cuda.synchronize()
        print("Model and evaluator preflight passed.")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_group_id = (
        f"xnli-hi-de-en-multiseed-train{TRAIN_SAMPLES_PER_LANGUAGE}"
        f"-test{TEST_SAMPLES_PER_LANGUAGE}-{run_timestamp}"
    )
    results_group_dir = PROJECT_ROOT / "phase1" / "results" / experiment_group_id
    checkpoints_group_dir = PROJECT_ROOT / "phase1" / "checkpoints" / experiment_group_id
    results_group_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_group_dir.mkdir(parents=True, exist_ok=False)
    seed_results = []

    for seed in SEEDS:
        set_seed(seed)
        seed_name = f"seed-{seed}"
        result_dir = results_group_dir / seed_name / "evaluation"
        checkpoint_dir = checkpoints_group_dir / seed_name
        trainer = make_trainer(seed, checkpoint_dir)
        trainer.load_model_and_tokenizer()
        trainer.model.print_trainable_parameters()

        result = run_sequential_experiment(
            trainer,
            task_order=LANGUAGES,
            train_datasets=train_sets_by_seed[seed],
            test_datasets=test_sets,
            experiment_id=f"{experiment_group_id}-{seed_name}",
            output_dir=result_dir,
            dataset_name=DATASET_ID,
            task_name="xnli",
            run_metadata={
                "phase": 1,
                "method": "sequential_finetuning_no_replay",
                "seed": seed,
                "seed_group": experiment_group_id,
                "test_split_seed": TEST_SPLIT_SEED,
                "model": MODEL_ID,
                "train_samples_per_language": TRAIN_SAMPLES_PER_LANGUAGE,
                "test_samples_per_language": TEST_SAMPLES_PER_LANGUAGE,
                "test_split": "test",
                "prompt_protocol": "phase1_b0_nli_prompt_candidate_label_log_likelihood",
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

    report = make_experiment_report(experiment_group_id, seed_results)
    report_path = results_group_dir / "multiseed_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()