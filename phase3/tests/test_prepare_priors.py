import csv
import json

import numpy as np
import pytest

from phase3.prepare_embedding_similarity import centroid_similarity
from phase3.prepare_grambank_distance import (
    build_distance_matrix,
    prepare as prepare_grambank,
    resolve_language_ids,
)
from phase3.prepare_urielplus_distance import write_matrix
from phase3.prepare_urielplus_distance import codes_for_languages


def test_centroid_similarity_is_symmetric_with_unit_diagonal():
    languages, matrix = centroid_similarity({
        "en": np.array([[1.0, 0.0], [1.0, 0.0]]),
        "hi": np.array([[0.0, 1.0], [0.0, 1.0]]),
    })

    assert languages == ["en", "hi"]
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)
    assert np.allclose(matrix, matrix.T)


def test_centroid_similarity_rejects_mismatched_embedding_dimensions():
    with pytest.raises(ValueError, match="same dimension"):
        centroid_similarity({
            "en": np.array([[1.0, 0.0]]),
            "hi": np.array([[1.0, 0.0, 0.0]]),
        })


def test_grambank_reports_missing_pilot_languages_without_fake_matrix(tmp_path):
    languages_path = tmp_path / "languages.csv"
    languages_path.write_text(
        "ID,Name,ISO639P3code\n"
        "stan1293,English,\n"
        "hind1269,Hindi,\n"
        "swah1253,Swahili,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not cover every requested language"):
        prepare_grambank(tmp_path / "out", languages_csv=languages_path)

    coverage = json.loads(
        (tmp_path / "out" / "grambank_coverage.json").read_text(encoding="utf-8")
    )
    assert coverage["missing_project_languages"] == ["es", "de"]
    assert coverage["matrix_generated"] is False
    assert not (tmp_path / "out" / "grambank_distance.csv").exists()


def test_grambank_coverage_is_checked_for_the_configured_ten_languages(tmp_path):
    languages_path = tmp_path / "languages.csv"
    languages_path.write_text(
        "ID,Name,ISO639P3code\n"
        "stan1293,English,\n"
        "hind1269,Hindi,\n"
        "swah1253,Swahili,\n"
        "arab1267,Arabic,\n"
        "urdu1239,Urdu,\n"
        "bulg1262,Bulgarian,\n"
        "fren1241,French,\n"
        "russ1263,Russian,\n",
        encoding="utf-8",
    )

    resolved, missing = resolve_language_ids(
        languages_path,
        ("en", "de", "es", "fr", "sw", "ar", "ur", "bg", "ru", "hi"),
    )

    assert sorted(resolved["ar"]) == ["arab1267"]
    assert sorted(resolved["ur"]) == ["urdu1239"]
    assert sorted(resolved["bg"]) == ["bulg1262"]
    assert sorted(resolved["fr"]) == ["fren1241"]
    assert missing == ["de", "es"]


def test_grambank_categorical_distance_uses_only_shared_non_missing_features(
    tmp_path,
):
    languages_path = tmp_path / "languages.csv"
    languages_path.write_text(
        "ID,Name,ISO639P3code\n"
        "lang-en,English,\n"
        "lang-hi,Hindi,\n"
        "lang-es,Spanish,\n"
        "lang-de,German,\n"
        "lang-sw,Swahili,\n",
        encoding="utf-8",
    )
    resolved, missing = resolve_language_ids(languages_path)
    assert missing == []

    values_path = tmp_path / "values.csv"
    with values_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["Language_ID", "Parameter_ID", "Value"])
        writer.writerows([
            ["lang-en", "feature-1", "A"],
            ["lang-en", "feature-2", "1"],
            ["lang-hi", "feature-1", "B"],
            ["lang-hi", "feature-2", "?"],
        ])

    selected = {language: resolved[language] for language in ("en", "hi")}
    distances, coverage = build_distance_matrix(values_path, selected)
    assert list(selected) == ["en", "hi"]
    assert distances[0][1] == pytest.approx(1.0)
    assert coverage["en"]["hi"] == 1
    assert distances[0][0] == 0.0


def test_urielplus_matrix_writer_records_sha256_and_iso3_codes(tmp_path):
    output = tmp_path / "uriel_syntactic_distance.csv"
    metadata = write_matrix(
        output,
        ["eng", "hin"],
        [[0.0, 0.25], [0.25, 0.0]],
        "1.3.3",
    )

    assert metadata["package_version"] == "1.3.3"
    assert metadata["matrix_sha256"]
    assert output.read_text(encoding="utf-8").splitlines()[0] == "language,eng,hin"
    sidecar = json.loads(
        (tmp_path / "uriel_syntactic_distance_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["matrix_sha256"] == metadata["matrix_sha256"]


def test_urielplus_resolves_all_v2_language_codes():
    assert codes_for_languages(
        ("en", "de", "es", "fr", "sw", "ar", "ur", "bg", "ru", "hi")
    ) == ["eng", "deu", "spa", "fra", "swa", "arb", "urd", "bul", "rus", "hin"]
