"""Create a Grambank distance matrix when all requested languages are covered."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
from collections.abc import Sequence
from typing import Any


GRAMBANK_RAW_ROOT = (
    "https://raw.githubusercontent.com/grambank/grambank/v1.0.3/cldf"
)
LANGUAGE_NAMES = {
    "en": ("English",),
    "hi": ("Hindi",),
    "es": ("Spanish",),
    "de": ("German",),
    "sw": ("Swahili",),
    "ar": ("Arabic", "Standard Arabic", "Modern Standard Arabic"),
    "ur": ("Urdu",),
    "bg": ("Bulgarian",),
    "ru": ("Russian",),
    "fr": ("French",),
    "tr": ("Turkish",),
    "th": ("Thai",),
    "zh": ("Chinese", "Mandarin", "Mandarin Chinese"),
}
ISO3_ALIASES = {
    "en": {"eng"},
    "hi": {"hin"},
    "es": {"spa"},
    "de": {"deu"},
    "sw": {"swa", "swh"},
    "ar": {"ara", "arb"},
    "ur": {"urd"},
    "bg": {"bul"},
    "ru": {"rus"},
    "fr": {"fra"},
    "tr": {"tur"},
    "th": {"tha"},
    "zh": {"zho", "chi", "cmn"},
}
DEFAULT_LANGUAGES = ("en", "hi", "es", "de", "sw")
MISSING_VALUES = {"", "?", "-", "NA", "N/A"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Phase3-prior-preparation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        destination.write_bytes(response.read())


def resolve_language_ids(
    languages_csv: str | Path,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve configured codes using ISO-639-3 codes or exact language names."""
    unknown = sorted(set(languages) - set(LANGUAGE_NAMES))
    if unknown:
        raise ValueError(f"No Grambank language mapping is configured for {unknown}.")
    if len(set(languages)) != len(languages):
        raise ValueError("Language codes must be unique.")
    resolved = {language: [] for language in languages}
    with Path(languages_csv).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"ID", "Name", "ISO639P3code"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Grambank languages.csv is missing required columns.")
        for row in reader:
            row_id = (row.get("ID") or "").strip()
            name = (row.get("Name") or "").strip().casefold()
            iso_code = (row.get("ISO639P3code") or "").strip().lower()
            if not row_id:
                continue
            for language, names in (
                (code, LANGUAGE_NAMES[code]) for code in languages
            ):
                if iso_code in ISO3_ALIASES[language] or name in {
                    item.casefold() for item in names
                }:
                    resolved[language].append(row_id)
    missing = [language for language, ids in resolved.items() if not ids]
    return resolved, missing


def _write_coverage(
    path: Path,
    resolved: dict[str, list[str]],
    missing: list[str],
    source_sha256: str,
    requested_languages: Sequence[str],
) -> None:
    report = {
        "dataset": "Grambank v1.0.3",
        "dataset_url": "https://doi.org/10.5281/zenodo.7740139",
        "requested_languages": list(requested_languages),
        "languages_csv_sha256": source_sha256,
        "resolved_language_ids": resolved,
        "missing_project_languages": missing,
        "matrix_generated": False,
    }
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_distance_matrix(
    values_csv: str | Path,
    resolved: dict[str, list[str]],
) -> tuple[list[list[float]], dict[str, dict[str, int]]]:
    """Compute mean categorical mismatch over shared Grambank features."""
    language_by_id = {
        row_id: language
        for language, row_ids in resolved.items()
        for row_id in row_ids
    }
    values: dict[str, dict[str, list[str]]] = {
        language: {} for language in resolved
    }
    with Path(values_csv).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Language_ID", "Parameter_ID", "Value"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Grambank values.csv is missing required columns.")
        for row in reader:
            language = language_by_id.get((row.get("Language_ID") or "").strip())
            feature = (row.get("Parameter_ID") or "").strip()
            value = (row.get("Value") or "").strip()
            if language is None or not feature or value in MISSING_VALUES:
                continue
            values[language].setdefault(feature, []).append(value)

    languages = list(resolved)
    matrix = [[0.0 for _ in languages] for _ in languages]
    coverage = {left: {} for left in languages}
    for left_index, left in enumerate(languages):
        for right_index in range(left_index + 1, len(languages)):
            right = languages[right_index]
            shared_features = sorted(set(values[left]) & set(values[right]))
            if not shared_features:
                raise ValueError(
                    f"No shared Grambank features for {left!r} and {right!r}."
                )
            disagreements = []
            for feature in shared_features:
                left_values = values[left][feature]
                right_values = values[right][feature]
                equal_pairs = sum(
                    left_value == right_value
                    for left_value in left_values
                    for right_value in right_values
                )
                total_pairs = len(left_values) * len(right_values)
                disagreements.append(1.0 - equal_pairs / total_pairs)
            distance = sum(disagreements) / len(disagreements)
            matrix[left_index][right_index] = distance
            matrix[right_index][left_index] = distance
            coverage[left][right] = len(shared_features)
            coverage[right][left] = len(shared_features)
        coverage[left][left] = len(values[left])
    return matrix, coverage


def _write_matrix(
    output_path: Path,
    languages: list[str],
    matrix: list[list[float]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["language", *languages])
        for language, row in zip(languages, matrix):
            writer.writerow([language, *row])


def prepare(
    output_dir: str | Path,
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    languages_csv: str | Path | None = None,
    values_csv: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    languages_path = (
        Path(languages_csv)
        if languages_csv
        else destination / "grambank_languages.csv"
    )
    if not languages_path.is_file():
        _download(
            f"{GRAMBANK_RAW_ROOT}/languages.csv",
            languages_path,
        )
    languages_hash = hashlib.sha256(languages_path.read_bytes()).hexdigest()
    resolved, missing = resolve_language_ids(languages_path, languages)
    coverage_path = destination / "grambank_coverage.json"
    _write_coverage(
        coverage_path, resolved, missing, languages_hash, languages
    )
    if missing:
        raise ValueError(
            "The pinned Grambank release does not cover every requested language: "
            f"{missing}. See {coverage_path}; do not impute or substitute a "
            "different language silently."
        )

    values_path = (
        Path(values_csv)
        if values_csv
        else destination / "grambank_values.csv"
    )
    if not values_path.is_file():
        _download(f"{GRAMBANK_RAW_ROOT}/values.csv", values_path)
    matrix, feature_coverage = build_distance_matrix(values_path, resolved)
    output_path = destination / "grambank_distance.csv"
    _write_matrix(output_path, list(resolved), matrix)
    metadata = {
        "dataset": "Grambank v1.0.3",
        "dataset_url": "https://doi.org/10.5281/zenodo.7740139",
        "distance_definition": (
            "mean categorical mismatch over shared non-missing features; "
            "multiple Grambank varieties with the same project language code "
            "are averaged by cross-variety pair frequency"
        ),
        "resolved_language_ids": resolved,
        "shared_feature_counts": feature_coverage,
        "languages_csv_sha256": languages_hash,
        "values_csv_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    coverage_report = {
        "dataset": "Grambank v1.0.3",
        "dataset_url": "https://doi.org/10.5281/zenodo.7740139",
        "languages_csv_sha256": languages_hash,
        "resolved_language_ids": resolved,
        "missing_project_languages": [],
        "matrix_generated": True,
        "matrix_sha256": metadata["matrix_sha256"],
    }
    coverage_path = destination / "grambank_coverage.json"
    coverage_path.write_text(
        json.dumps(coverage_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "grambank_distance_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Grambank distances if the selected languages are covered."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3") / "prior_inputs",
    )
    parser.add_argument("--languages-csv", type=Path)
    parser.add_argument("--values-csv", type=Path)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Configured project language codes to check and include.",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.output_dir,
        languages=args.languages,
        languages_csv=args.languages_csv,
        values_csv=args.values_csv,
    ), indent=2))


if __name__ == "__main__":
    main()
