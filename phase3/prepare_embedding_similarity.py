"""Generate a reproducible XNLI embedding-similarity prior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def centroid_similarity(
    embeddings_by_language: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Return cosine similarities between normalized language centroids."""
    languages = list(embeddings_by_language)
    if not languages:
        raise ValueError("At least one language embedding array is required.")
    centroids = []
    feature_count = None
    for language in languages:
        embeddings = np.asarray(embeddings_by_language[language], dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"{language!r} embeddings must be a non-empty 2-D array.")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{language!r} embeddings contain non-finite values.")
        if feature_count is None:
            feature_count = embeddings.shape[1]
        elif embeddings.shape[1] != feature_count:
            raise ValueError("All language embeddings must have the same dimension.")
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise ValueError(f"{language!r} has a zero-norm embedding centroid.")
        centroids.append(centroid / norm)

    similarity = np.clip(np.asarray(centroids) @ np.asarray(centroids).T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return languages, similarity


def _read_language_texts(
    prepared_data_dir: str | Path,
    languages: tuple[str, ...],
) -> dict[str, list[str]]:
    data_dir = Path(prepared_data_dir) / "data"
    texts: dict[str, list[str]] = {}
    for language in languages:
        path = data_dir / f"{language}.train.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared XNLI training split not found: {path}")
        language_texts = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                example = record.get("example", record)
                premise = example.get("premise")
                hypothesis = example.get("hypothesis")
                if not isinstance(premise, str) or not isinstance(hypothesis, str):
                    raise ValueError(
                        f"Expected premise and hypothesis strings in {path}."
                    )
                language_texts.append(
                    f"Premise: {premise}\nHypothesis: {hypothesis}"
                )
        if not language_texts:
            raise ValueError(f"No training examples found in {path}.")
        texts[language] = language_texts
    return texts


def _write_matrix(
    output_path: Path,
    languages: list[str],
    matrix: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["language", *languages])
        for language, row in zip(languages, matrix):
            writer.writerow([language, *[float(value) for value in row]])


def prepare(
    protocol_path: str | Path,
    prepared_data_dir: str | Path,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    """Prepare the fixed Phase 2 sample and encode it without model training."""
    from phase2.prepare_data import prepare_xnli
    from phase2.protocol import load_protocol
    from phase2.run_baseline import validate_prepared_artifacts

    protocol = load_protocol(protocol_path)
    protocol_languages = tuple(protocol["dataset"]["languages"])
    prepared_dir = Path(prepared_data_dir)
    if not (prepared_dir / "protocol_manifest.json").is_file():
        prepare_xnli(protocol_path, prepared_dir)
    data_manifest = validate_prepared_artifacts(protocol, prepared_dir)
    texts_by_language = _read_language_texts(prepared_dir, protocol_languages)

    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    import torch

    resolved_revision = model_info(model_name).sha
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_name,
        revision=resolved_revision,
        device=device,
    )
    model.max_seq_length = 256
    embeddings = {
        language: model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for language, texts in texts_by_language.items()
    }
    languages, matrix = centroid_similarity(embeddings)
    output = Path(output_path)
    _write_matrix(output, languages, matrix)
    metadata = {
        "source_dataset": "XNLI training split prepared by Phase 2 protocol",
        "protocol_manifest_sha256": hashlib.sha256(
            (prepared_dir / "protocol_manifest.json").read_bytes()
        ).hexdigest(),
        "data_manifest_sha256": data_manifest["data_manifest_sha256"],
        "languages": list(languages),
        "examples_per_language": {
            language: len(texts_by_language[language])
            for language in languages
        },
        "text_representation": "Premise: {premise}\\nHypothesis: {hypothesis}",
        "encoder": model_name,
        "encoder_revision": resolved_revision,
        "sentence_transformers_version": importlib.metadata.version(
            "sentence-transformers"
        ),
        "max_sequence_length": model.max_seq_length,
        "batch_size": batch_size,
        "device": device,
        "language_representation": (
            "L2-normalized mean of L2-normalized sentence-pair embeddings"
        ),
        "similarity": "cosine similarity between language centroids",
        "matrix_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "matrix_path": output.name,
    }
    metadata_path = output.with_name("embedding_similarity_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 3 XNLI embedding-similarity matrix."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("phase2") / "configs" / "pilot_protocol.json",
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        default=Path("phase3") / "embedding_xnli_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_similarity.csv"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    metadata = prepare(
        args.protocol,
        args.prepared_data_dir,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
