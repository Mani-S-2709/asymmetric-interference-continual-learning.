"""Reusable NLI formatting and deterministic dataset selection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import random
from numbers import Integral
from typing import Any, Sequence

from continual_learning.stream_loader import (
    HuggingFaceStreamLoader,
    StreamStage,
)


NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_LABEL_IDS = {label: index for index, label in enumerate(NLI_LABELS)}


def format_nli_example(example: Mapping[str, Any]) -> dict[str, str]:
    """Format a three-way NLI example for the Phase 1 candidate-label task."""
    missing = {"premise", "hypothesis", "label"} - example.keys()
    if missing:
        raise ValueError(
            f"NLI example is missing required fields: {sorted(missing)}."
        )

    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]
    if not isinstance(premise, str) or not premise.strip():
        raise TypeError("NLI premise must be a non-empty string.")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise TypeError("NLI hypothesis must be a non-empty string.")
    if isinstance(label, bool):
        raise TypeError("NLI label cannot be boolean.")
    if isinstance(label, Integral):
        label_id = int(label)
    elif isinstance(label, str):
        label_name = label.strip().lower()
        if label_name not in NLI_LABEL_IDS:
            raise ValueError(f"Unknown NLI label: {label!r}.")
        label_id = NLI_LABEL_IDS[label_name]
    else:
        raise TypeError("NLI label must be an integer ID or known label name.")
    if label_id not in range(len(NLI_LABELS)):
        raise ValueError(f"NLI label ID must be 0, 1, or 2; received {label_id}.")

    prompt = (
        "Task: Natural language inference\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Choose one label: entailment, neutral, contradiction\n"
        "Answer:"
    )
    return {"prompt": prompt, "target": f" {NLI_LABELS[label_id]}"}


def select_source_indices(size: int, count: int, seed: int) -> list[int]:
    """Select a reproducible random subset and return its source row indices."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer.")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if count > size:
        raise ValueError(f"Requested {count} examples from a dataset of size {size}.")
    return random.Random(seed).sample(range(size), count)


def selection_sha256(indices: list[int]) -> str:
    """Hash the exact ordered source-row selection for run provenance."""
    payload = json.dumps(indices, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def select_dataset_rows(dataset: Any, indices: list[int]) -> Any:
    """Apply selected source-row indices to a Hugging Face-style dataset."""
    if not callable(getattr(dataset, "select", None)):
        raise TypeError("Dataset must provide a select(indices) method.")
    return dataset.select(indices)


def prepare_nli_stream(
    stages: Sequence[StreamStage],
    loader: HuggingFaceStreamLoader,
    *,
    experiment_id: str,
    seeds: Sequence[int],
    train_count: int,
    test_count: int,
    test_seed: int,
    train_split: str = "train",
    test_split: str = "test",
    require_parallel_label_alignment: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load, validate, and select reproducible NLI data for a language stream."""
    if not stages or len({stage.language for stage in stages}) != len(stages):
        raise ValueError("stages must contain unique language codes.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique training seeds.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string.")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in (train_count, test_count)
    ):
        raise ValueError("train_count and test_count must be positive integers.")
    if any(
        not isinstance(split, str) or not split.strip()
        for split in (train_split, test_split)
    ):
        raise ValueError("train_split and test_split must be non-empty strings.")

    languages = [stage.language for stage in stages]
    train_by_seed = {seed: {} for seed in seeds}
    test_by_language = {}
    source_manifest: dict[str, Any] = {}
    train_labels_by_seed: dict[int, dict[str, list[str]]] = {
        seed: {} for seed in seeds
    }
    train_indices_by_seed: dict[int, dict[str, list[int]]] = {
        seed: {} for seed in seeds
    }
    test_labels_by_language: dict[str, list[str]] = {}
    test_indices_by_language: dict[str, list[int]] = {}

    for stage in stages:
        train_raw = loader.load_stage(replace(stage, split=train_split))
        test_raw = loader.load_stage(replace(stage, split=test_split))
        if len(train_raw) < train_count:
            raise ValueError(
                f"{stage.language}: requested {train_count} train examples, "
                f"but source has {len(train_raw)}."
            )
        if len(test_raw) < test_count:
            raise ValueError(
                f"{stage.language}: requested {test_count} test examples, "
                f"but source has {len(test_raw)}."
            )

        test_indices = select_source_indices(len(test_raw), test_count, test_seed)
        test_subset = select_dataset_rows(test_raw, test_indices)
        test_labels = []
        for index in range(len(test_subset)):
            try:
                test_labels.append(
                    format_nli_example(test_subset[index])["target"]
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"Invalid {test_split} NLI example {index} for "
                    f"{stage.language!r}: {exc}"
                ) from exc
        test_by_language[stage.language] = test_subset
        test_labels_by_language[stage.language] = test_labels
        test_indices_by_language[stage.language] = test_indices

        source_train_indices = {}
        for seed in seeds:
            train_indices = select_source_indices(len(train_raw), train_count, seed)
            train_subset = select_dataset_rows(train_raw, train_indices)
            formatted = train_subset.map(
                format_nli_example,
                remove_columns=train_subset.column_names,
                desc=f"Formatting {stage.language} NLI training data",
            )
            if not all(
                set(row) == {"prompt", "target"}
                for row in formatted.select(range(min(3, len(formatted))))
            ):
                raise ValueError(
                    f"Formatted training data for {stage.language!r} "
                    "has unexpected columns."
                )
            train_by_seed[seed][stage.language] = formatted
            train_labels_by_seed[seed][stage.language] = [
                row["target"] for row in formatted
            ]
            train_indices_by_seed[seed][stage.language] = train_indices
            source_train_indices[str(seed)] = {
                "count": len(train_indices),
                "indices_sha256": selection_sha256(train_indices),
                "indices": train_indices,
            }

        source_manifest[stage.language] = {
            "dataset_id": stage.dataset_id,
            "dataset_config": stage.dataset_config,
            "revision": stage.revision,
            "train_split": train_split,
            "test_split": test_split,
            "train_source_rows": len(train_raw),
            "test_source_rows": len(test_raw),
            "train_fingerprint": getattr(train_raw, "_fingerprint", None),
            "test_fingerprint": getattr(test_raw, "_fingerprint", None),
            "train_selection_by_seed": source_train_indices,
            "test_selection": {
                "seed": test_seed,
                "count": len(test_indices),
                "indices_sha256": selection_sha256(test_indices),
                "indices": test_indices,
            },
        }

    if require_parallel_label_alignment:
        reference_language = languages[0]
        for language in languages[1:]:
            if (
                test_indices_by_language[language]
                != test_indices_by_language[reference_language]
                or test_labels_by_language[language]
                != test_labels_by_language[reference_language]
            ):
                raise ValueError(
                    "Selected test rows are not parallel-label aligned "
                    f"between {reference_language!r} and {language!r}."
                )
            for seed in seeds:
                if (
                    train_indices_by_seed[seed][language]
                    != train_indices_by_seed[seed][reference_language]
                    or train_labels_by_seed[seed][language]
                    != train_labels_by_seed[seed][reference_language]
                ):
                    raise ValueError(
                        "Selected training rows are not parallel-label "
                        f"aligned for seed {seed} between "
                        f"{reference_language!r} and {language!r}."
                    )

    data_manifest = {
        "experiment_id": experiment_id,
        "task_order": languages,
        "seeds": list(seeds),
        "selection_method": "random.Random(seed).sample(range(source_rows), count)",
        "require_parallel_label_alignment": require_parallel_label_alignment,
        "sources": source_manifest,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(
            data_manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    data_manifest["data_manifest_sha256"] = manifest_hash
    return train_by_seed, test_by_language, data_manifest
