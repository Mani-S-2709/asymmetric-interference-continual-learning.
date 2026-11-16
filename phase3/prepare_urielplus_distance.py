"""Generate the five-language URIEL+ syntactic-distance CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


PILOT_LANGUAGE_CODES = {
    "en": "eng",
    "hi": "hin",
    "es": "spa",
    "de": "deu",
    "sw": "swa",
}


def generate_matrix() -> tuple[list[str], list[list[float]], str]:
    """Query the pinned URIEL+ package and validate its distance matrix."""
    import numpy as np
    from urielplus import urielplus

    version = importlib.metadata.version("urielplus")
    if version != "1.3.3":
        raise RuntimeError(
            f"Expected urielplus 1.3.3; found {version}. "
            "Pin the validated package version for reproducibility."
        )
    languages = list(PILOT_LANGUAGE_CODES.values())
    raw_matrix = np.asarray(
        urielplus.URIELPlus().new_distance("syntactic", languages),
        dtype=np.float64,
    )
    if raw_matrix.shape != (len(languages), len(languages)):
        raise ValueError(f"Unexpected URIEL+ matrix shape: {raw_matrix.shape}.")
    if not np.isfinite(raw_matrix).all():
        raise ValueError("URIEL+ returned non-finite distances.")
    if (raw_matrix < 0).any():
        raise ValueError("URIEL+ syntactic distances must be non-negative.")
    if not np.allclose(raw_matrix, raw_matrix.T, atol=1e-12, rtol=0):
        raise ValueError("URIEL+ syntactic distances are not symmetric.")
    if not np.allclose(np.diag(raw_matrix), 0.0, atol=1e-12, rtol=0):
        raise ValueError("URIEL+ syntactic distance diagonal must be zero.")
    return languages, raw_matrix.tolist(), version


def write_matrix(
    output_path: str | Path,
    languages: list[str],
    matrix: list[list[float]],
    package_version: str,
) -> dict[str, Any]:
    """Write the labeled matrix and a provenance record."""
    if len(set(languages)) != len(languages) or len(matrix) != len(languages):
        raise ValueError("Matrix labels must be unique and match the matrix size.")
    if any(len(row) != len(languages) for row in matrix):
        raise ValueError("The URIEL+ matrix must be square.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["language", *languages])
        for language, row in zip(languages, matrix):
            writer.writerow([language, *row])
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "source": "URIEL+ Python package",
        "source_url": "https://github.com/Masonshipton25/URIELPlus",
        "distance": "syntactic",
        "package_version": package_version,
        "language_codes": {
            project_code: matrix_code
            for project_code, matrix_code in PILOT_LANGUAGE_CODES.items()
            if matrix_code in languages
        },
        "matrix_sha256": digest,
        "matrix_path": output.name,
    }
    metadata_path = output.with_name("uriel_syntactic_distance_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 2 pilot URIEL+ syntactic-distance matrix."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("uriel_syntactic_distance.csv"),
    )
    args = parser.parse_args()
    languages, matrix, version = generate_matrix()
    metadata = write_matrix(args.output, languages, matrix, version)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
