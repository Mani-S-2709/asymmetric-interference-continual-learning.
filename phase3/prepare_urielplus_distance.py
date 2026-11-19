"""Generate a URIEL+ syntactic-distance CSV for configured XNLI languages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any


URIEL_LANGUAGE_CODES = {
    "en": "eng",
    "hi": "hin",
    "es": "spa",
    "de": "deu",
    "sw": "swa",
    "fr": "fra",
    "ar": "arb",
    "ur": "urd",
    "bg": "bul",
    "ru": "rus",
    "tr": "tur",
    "th": "tha",
    "zh": "cmn",
}
DEFAULT_LANGUAGES = ("en", "hi", "es", "de", "sw")


def codes_for_languages(languages: Sequence[str]) -> list[str]:
    """Resolve XNLI codes to the URIEL+ ISO 639-3 identifiers."""
    unknown = sorted(set(languages) - set(URIEL_LANGUAGE_CODES))
    if unknown:
        raise ValueError(f"No URIEL+ language-code mapping is configured for {unknown}.")
    if len(set(languages)) != len(languages):
        raise ValueError("Language codes must be unique.")
    return [URIEL_LANGUAGE_CODES[language] for language in languages]


def generate_matrix(
    languages: Sequence[str] = DEFAULT_LANGUAGES,
) -> tuple[list[str], list[list[float]], str]:
    """Query the pinned URIEL+ package and validate its distance matrix."""
    import numpy as np
    from urielplus import urielplus

    version = importlib.metadata.version("urielplus")
    if version != "1.3.3":
        raise RuntimeError(
            f"Expected urielplus 1.3.3; found {version}. "
            "Pin the validated package version for reproducibility."
        )
    uriel_codes = codes_for_languages(languages)
    raw_matrix = np.asarray(
        urielplus.URIELPlus().new_distance("syntactic", uriel_codes),
        dtype=np.float64,
    )
    if raw_matrix.shape != (len(uriel_codes), len(uriel_codes)):
        raise ValueError(f"Unexpected URIEL+ matrix shape: {raw_matrix.shape}.")
    if not np.isfinite(raw_matrix).all():
        raise ValueError("URIEL+ returned non-finite distances.")
    if (raw_matrix < 0).any():
        raise ValueError("URIEL+ syntactic distances must be non-negative.")
    if not np.allclose(raw_matrix, raw_matrix.T, atol=1e-12, rtol=0):
        raise ValueError("URIEL+ syntactic distances are not symmetric.")
    if not np.allclose(np.diag(raw_matrix), 0.0, atol=1e-12, rtol=0):
        raise ValueError("URIEL+ syntactic distance diagonal must be zero.")
    return uriel_codes, raw_matrix.tolist(), version


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
            for project_code, matrix_code in URIEL_LANGUAGE_CODES.items()
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
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="XNLI language codes to include in the matrix.",
    )
    args = parser.parse_args()
    languages, matrix, version = generate_matrix(args.languages)
    metadata = write_matrix(args.output, languages, matrix, version)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
