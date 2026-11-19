"""Prepare a target-language interference matrix with explicit k-NN imputation."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from phase3.prior_analysis import LANGUAGE_METADATA, load_wide_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MATRIX = (
    PROJECT_ROOT
    / "phase3"
    / "results"
    / "rq1_kaggle_2026-11-16"
    / "priors"
    / "interference_matrix.csv"
)
PHASE4_LANGUAGES = ("en", "de", "es", "fr", "sw", "ar", "ur", "bg", "ru", "hi")
URIEL_CODE_ALIASES = {"swh": "swa"}
EXPECTED_URIELPLUS_VERSION = "1.3.3"


def impute_missing_language(
    scores: Mapping[str, Mapping[str, float]],
    distances: Mapping[str, float],
    missing_language: str,
    *,
    k: int = 5,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Impute one directed score row and column from its k nearest languages."""
    if not isinstance(missing_language, str) or not missing_language:
        raise ValueError("missing_language must be a non-empty string.")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be positive.")

    languages = set(scores)
    if not languages or missing_language in languages:
        raise ValueError("scores must contain known languages but not the missing one.")
    if any(set(row) != languages for row in scores.values()):
        raise ValueError("scores must be a square matrix with matching language codes.")
    if k > len(languages):
        raise ValueError(f"k cannot exceed the {len(languages)} known languages.")

    normalized_scores: dict[str, dict[str, float]] = {}
    for past_language, row in scores.items():
        normalized_scores[past_language] = {}
        for incoming_language, score in row.items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("Interference scores must be finite numbers.")
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("Interference scores must be finite numbers.")
            normalized_scores[past_language][incoming_language] = value

    unknown_distance_languages = set(distances) - languages
    if unknown_distance_languages:
        raise ValueError(
            "URIEL+ distances contain unknown matrix languages: "
            f"{sorted(unknown_distance_languages)}."
        )
    neighbor_distances = {}
    for language, distance in distances.items():
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError(f"URIEL+ distance for {language!r} must be finite.")
        distance_value = float(distance)
        if not math.isfinite(distance_value) or distance_value < 0:
            raise ValueError(f"URIEL+ distance for {language!r} must be non-negative.")
        neighbor_distances[language] = distance_value
    if len(neighbor_distances) < k:
        raise ValueError(
            f"Only {len(neighbor_distances)} of {len(languages)} known languages "
            f"have valid URIEL+ distances; k={k} requires at least {k}."
        )

    neighbors = sorted(
        neighbor_distances,
        key=lambda language: (neighbor_distances[language], language),
    )[:k]
    expanded = {
        language: dict(row) for language, row in normalized_scores.items()
    }
    expanded[missing_language] = {
        incoming_language: sum(
            normalized_scores[neighbor][incoming_language]
            for neighbor in neighbors
        ) / k
        for incoming_language in languages
    }
    for past_language in languages:
        expanded[past_language][missing_language] = sum(
            normalized_scores[past_language][neighbor]
            for neighbor in neighbors
        ) / k
    expanded[missing_language][missing_language] = 0.0

    provenance = [
        {
            "matrix_language": language,
            "uriel_syntactic_distance": neighbor_distances[language],
        }
        for language in neighbors
    ]
    return expanded, provenance


def _uriel_code(matrix_language: str) -> str:
    base_code = matrix_language.split("_", maxsplit=1)[0]
    return URIEL_CODE_ALIASES.get(base_code, base_code)


def generate_arabic_distances(
    source_languages: list[str],
) -> tuple[dict[str, float], str, dict[str, str]]:
    """Query the pinned URIEL+ package for Arabic-to-source distances."""
    if len(set(source_languages)) != len(source_languages):
        raise ValueError("Source matrix language codes must be unique.")

    uriel_codes = {
        language: _uriel_code(language) for language in source_languages
    }
    missing_code = LANGUAGE_METADATA["ar"]["iso3_code"]
    query_codes = sorted(set(uriel_codes.values()) | {missing_code})

    from urielplus import urielplus

    package_version = importlib.metadata.version("urielplus")
    if package_version != EXPECTED_URIELPLUS_VERSION:
        raise RuntimeError(
            f"Expected urielplus {EXPECTED_URIELPLUS_VERSION}; found "
            f"{package_version}. Pin the validated package version."
        )
    raw_distances = np.asarray(
        urielplus.URIELPlus().new_distance("syntactic", query_codes),
        dtype=np.float64,
    )
    if raw_distances.shape != (len(query_codes), len(query_codes)):
        raise ValueError(
            f"Unexpected URIEL+ distance matrix shape: {raw_distances.shape}."
        )
    if not np.isfinite(raw_distances).all() or (raw_distances < -1).any():
        raise ValueError("URIEL+ returned non-finite or invalid syntactic distances.")
    code_indexes = {code: index for index, code in enumerate(query_codes)}
    distances = {
        language: float(raw_distances[code_indexes[missing_code], code_indexes[code]])
        for language, code in uriel_codes.items()
        if raw_distances[code_indexes[missing_code], code_indexes[code]] >= 0
    }
    return distances, package_version, uriel_codes


def prepare_phase4_matrix(
    source_path: str | Path = DEFAULT_SOURCE_MATRIX,
    *,
    k: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Load published scores and construct the selected ten-language matrix."""
    if isinstance(k, bool) or not isinstance(k, int) or k != 5:
        raise ValueError("The frozen Phase 4 Arabic imputation requires k=5.")
    source_path = Path(source_path)
    source_scores = load_wide_matrix(source_path)
    if set(PHASE4_LANGUAGES) - set(LANGUAGE_METADATA):
        raise ValueError("Phase 4 language metadata is incomplete.")

    target_codes = {
        language: LANGUAGE_METADATA[language]["matrix_code"]
        for language in PHASE4_LANGUAGES
    }
    missing_codes = sorted(set(target_codes.values()) - set(source_scores))
    expected_missing_code = LANGUAGE_METADATA["ar"]["matrix_code"]
    if missing_codes != [expected_missing_code]:
        raise ValueError(
            "The source matrix must cover every Phase 4 language except Arabic; "
            f"missing codes: {missing_codes}."
        )

    distances, package_version, uriel_codes = generate_arabic_distances(
        list(source_scores)
    )
    expanded_scores, neighbors = impute_missing_language(
        source_scores,
        distances,
        expected_missing_code,
        k=k,
    )
    target_scores = {
        past_language: {
            incoming_language: (
                0.0
                if past_language == incoming_language
                else expanded_scores[target_codes[past_language]][
                    target_codes[incoming_language]
                ]
            )
            for incoming_language in PHASE4_LANGUAGES
        }
        for past_language in PHASE4_LANGUAGES
    }
    metadata = {
        "source_matrix": source_path.name,
        "source_matrix_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "score_semantics": (
            "Rows are past/affected languages; columns are incoming languages; "
            "negative scores indicate interference."
        ),
        "imputation": {
            "language": "ar",
            "method": "uniform_mean_of_k_nearest_languages",
            "distance": "URIEL+ syntactic",
            "distance_metric": "angular",
            "feature_aggregation": "U",
            "k": k,
            "package": "urielplus",
            "package_version": package_version,
            "source_url": "https://github.com/Masonshipton25/URIELPlus",
            "neighbors": neighbors,
            "candidate_languages_with_valid_distance": len(distances),
            "excluded_languages_without_valid_distance": sorted(
                set(source_scores) - set(distances)
            ),
        },
        "language_codes": {
            language: {
                "matrix_code": target_codes[language],
                "uriel_code": LANGUAGE_METADATA[language]["iso3_code"],
            }
            for language in PHASE4_LANGUAGES
        },
    }
    metadata["uriel_source_codes"] = uriel_codes
    return target_scores, metadata


def write_phase4_matrix(
    scores: Mapping[str, Mapping[str, float]],
    metadata: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the selected matrix and a checksummed imputation provenance file."""
    languages = list(PHASE4_LANGUAGES)
    if set(scores) != set(languages) or any(
        set(row) != set(languages) for row in scores.values()
    ):
        raise ValueError("Phase 4 interference scores must be a complete 10x10 matrix.")
    for row in scores.values():
        for score in row.values():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("Phase 4 interference scores must be finite numbers.")
            if not math.isfinite(float(score)):
                raise ValueError("Phase 4 interference scores must be finite numbers.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["language", *languages])
        for language in languages:
            writer.writerow([language, *(scores[language][other] for other in languages)])
    result_metadata = dict(metadata)
    result_metadata["matrix_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    result_metadata["matrix_path"] = output.name
    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(result_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_metadata


def load_phase4_matrix(path: str | Path) -> dict[str, dict[str, float]]:
    """Read a prepared Phase 4 CSV and require its matching provenance record."""
    matrix_path = Path(path)
    metadata_path = matrix_path.with_name(f"{matrix_path.stem}_metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Matrix provenance file not found: {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    if metadata.get("matrix_sha256") != digest:
        raise ValueError("Phase 4 interference matrix checksum does not match provenance.")
    matrix = load_wide_matrix(matrix_path)
    if set(matrix) != set(PHASE4_LANGUAGES):
        raise ValueError("Prepared Phase 4 matrix has unexpected language coverage.")
    if any(matrix[language][language] != 0.0 for language in PHASE4_LANGUAGES):
        raise ValueError("Prepared Phase 4 matrix must have a zero diagonal.")
    imputation = metadata.get("imputation")
    if (
        not isinstance(imputation, Mapping)
        or imputation.get("method") != "uniform_mean_of_k_nearest_languages"
        or imputation.get("distance") != "URIEL+ syntactic"
        or imputation.get("distance_metric") != "angular"
        or imputation.get("feature_aggregation") != "U"
        or imputation.get("k") != 5
        or imputation.get("language") != "ar"
        or imputation.get("package") != "urielplus"
        or imputation.get("package_version") != EXPECTED_URIELPLUS_VERSION
        or not isinstance(imputation.get("neighbors"), list)
        or len(imputation["neighbors"]) != 5
        or not isinstance(imputation.get("candidate_languages_with_valid_distance"), int)
        or imputation["candidate_languages_with_valid_distance"] < 5
        or not isinstance(metadata.get("source_matrix_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", metadata["source_matrix_sha256"]) is None
    ):
        raise ValueError("Phase 4 matrix provenance does not match the frozen k=5 rule.")
    neighbor_codes = []
    for neighbor in imputation["neighbors"]:
        if not isinstance(neighbor, Mapping):
            raise ValueError("Phase 4 imputation neighbor metadata is invalid.")
        language = neighbor.get("matrix_language")
        distance = neighbor.get("uriel_syntactic_distance")
        if (
            not isinstance(language, str)
            or not language
            or isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(distance))
            or distance < 0
        ):
            raise ValueError("Phase 4 imputation neighbor metadata is invalid.")
        neighbor_codes.append(language)
    if len(set(neighbor_codes)) != 5:
        raise ValueError("Phase 4 imputation neighbors must be unique.")
    return {
        past_language: {
            incoming_language: matrix[past_language][incoming_language]
            for incoming_language in PHASE4_LANGUAGES
        }
        for past_language in PHASE4_LANGUAGES
    }
