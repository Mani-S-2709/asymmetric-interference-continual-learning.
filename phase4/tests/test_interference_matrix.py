import csv

import pytest

import phase4.interference_matrix as interference_matrix
from phase4.interference_matrix import (
    impute_missing_language,
    load_phase4_matrix,
    prepare_phase4_matrix,
    write_phase4_matrix,
)


def test_impute_missing_language_uses_same_neighbors_for_directed_row_and_column():
    scores = {
        "a": {"a": 0.0, "b": -2.0, "c": 4.0},
        "b": {"a": -4.0, "b": 0.0, "c": 6.0},
        "c": {"a": 8.0, "b": 10.0, "c": 0.0},
    }

    expanded, neighbors = impute_missing_language(
        scores,
        {"a": 0.1, "b": 0.2, "c": 0.3},
        "missing",
        k=2,
    )

    assert [neighbor["matrix_language"] for neighbor in neighbors] == ["a", "b"]
    assert expanded["missing"] == {
        "a": -2.0,
        "b": -1.0,
        "c": 5.0,
        "missing": 0.0,
    }
    assert expanded["a"]["missing"] == -1.0
    assert expanded["b"]["missing"] == -2.0
    assert expanded["a"]["b"] == scores["a"]["b"]


def test_imputation_ignores_languages_without_a_valid_distance():
    scores = {
        "a": {"a": 0.0, "b": -2.0, "c": 4.0},
        "b": {"a": -4.0, "b": 0.0, "c": 6.0},
        "c": {"a": 8.0, "b": 10.0, "c": 0.0},
    }

    _, neighbors = impute_missing_language(
        scores,
        {"a": 0.1, "b": 0.2},
        "missing",
        k=2,
    )

    assert [neighbor["matrix_language"] for neighbor in neighbors] == ["a", "b"]


@pytest.mark.parametrize(
    ("distances", "k", "message"),
    [
        ({"a": 0.1}, 2, "Only 1 of 2 known languages"),
        ({"a": 0.1, "b": -0.1}, 1, "non-negative"),
        ({"a": 0.1, "b": 0.2}, 3, "cannot exceed"),
        ({"a": 0.1, "b": 0.2}, 0, "k must be positive"),
    ],
)
def test_imputation_rejects_invalid_neighbors(distances, k, message):
    scores = {"a": {"a": 0.0, "b": 1.0}, "b": {"a": 2.0, "b": 0.0}}

    with pytest.raises((TypeError, ValueError), match=message):
        impute_missing_language(scores, distances, "missing", k=k)


def test_written_matrix_requires_matching_provenance(tmp_path):
    languages = ("en", "de", "es", "fr", "sw", "ar", "ur", "bg", "ru", "hi")
    scores = {
        language: {other: 0.0 for other in languages}
        for language in languages
    }
    metadata = {
        "source_matrix_sha256": "a" * 64,
        "imputation": {
            "language": "ar",
            "method": "uniform_mean_of_k_nearest_languages",
            "distance": "URIEL+ syntactic",
            "distance_metric": "angular",
            "feature_aggregation": "U",
            "k": 5,
            "package": "urielplus",
            "package_version": "1.3.3",
            "candidate_languages_with_valid_distance": 5,
            "neighbors": [
                {
                    "matrix_language": str(index),
                    "uriel_syntactic_distance": index / 10,
                }
                for index in range(5)
            ],
        }
    }
    output = tmp_path / "matrix.csv"

    write_phase4_matrix(scores, metadata, output)

    assert set(load_phase4_matrix(output)) == set(languages)
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_phase4_matrix(output)


def test_prepare_phase4_matrix_maps_nine_source_languages_and_imputes_arabic(
    tmp_path,
    monkeypatch,
):
    languages = ("en", "de", "es", "fr", "sw", "ur", "bg", "ru", "hi")
    matrix_codes = [
        interference_matrix.LANGUAGE_METADATA[language]["matrix_code"]
        for language in languages
    ]
    source_path = tmp_path / "source.csv"
    with source_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["language", *matrix_codes])
        for row_index, row_language in enumerate(matrix_codes):
            writer.writerow([
                row_language,
                *[
                    float(row_index * 10 + column_index)
                    for column_index in range(len(matrix_codes))
                ],
            ])
    monkeypatch.setattr(
        interference_matrix,
        "generate_arabic_distances",
        lambda source_languages: (
            {language: float(index) for index, language in enumerate(source_languages)},
            "1.3.3",
            {language: language.split("_", maxsplit=1)[0] for language in source_languages},
        ),
    )

    scores, metadata = prepare_phase4_matrix(source_path, k=5)

    assert set(scores) == set(interference_matrix.PHASE4_LANGUAGES)
    assert len(metadata["imputation"]["neighbors"]) == 5
    assert metadata["imputation"]["language"] == "ar"
    assert scores["ar"]["ar"] == 0.0
    assert scores["en"]["fr"] == float(
        matrix_codes.index("eng_Latn") * 10
        + matrix_codes.index("fra_Latn")
    )
