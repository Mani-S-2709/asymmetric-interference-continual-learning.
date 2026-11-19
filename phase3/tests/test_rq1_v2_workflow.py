import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from phase2.protocol import build_manifest, load_protocol
from phase3.rq1_v2_workflow import (
    INPUT_SUMMARY_NAME,
    PROTOCOL_ARCHIVE_NAME,
    _load_input_bundle,
    _validated_interference_matrix,
    _write_imputed_interference_prior,
)
from phase3.prior_analysis import LANGUAGE_METADATA, load_wide_matrix
from phase4 import interference_matrix as phase4_interference_matrix


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "rq1_v2_10lang_protocol.json"
)


def _write_input_bundle(
    path: Path,
    *,
    bad_matrix_hash: bool = False,
    stale_protocol: bool = False,
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    if stale_protocol:
        protocol["protocol_id"] = "xnli-rq1-v2-10lang-500x200-v1"
    protocol_manifest = build_manifest(protocol)
    matrix = b"language,en\n"
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_manifest["protocol_sha256"],
        "data_manifest_sha256": "a" * 64,
        "languages": protocol["dataset"]["languages"],
        "orders": [order["id"] for order in protocol["orders"]],
        "seeds": protocol["training"]["seeds"],
        "prior_matrix_sha256": {
            "priors/example.csv": (
                "0" * 64
                if bad_matrix_hash
                else hashlib.sha256(matrix).hexdigest()
            )
        },
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            INPUT_SUMMARY_NAME,
            json.dumps(summary),
        )
        archive.writestr(
            PROTOCOL_ARCHIVE_NAME,
            json.dumps(protocol),
        )
        archive.writestr("priors/example.csv", matrix)


def test_v2_protocol_covers_every_directed_language_pair():
    protocol = load_protocol(PROTOCOL_PATH)
    observed_pairs = {
        (past, incoming)
        for order in protocol["orders"]
        for index, past in enumerate(order["languages"])
        for incoming in order["languages"][index + 1:]
    }
    language_count = len(protocol["dataset"]["languages"])

    assert protocol["dataset"]["languages"] == [
        "en", "de", "es", "fr", "sw", "ar", "ur", "bg", "ru", "hi"
    ]
    assert protocol["training"]["seeds"] == [42, 43, 44]
    assert len(protocol["orders"]) == 5
    assert len(observed_pairs) == language_count * (language_count - 1)


def test_input_bundle_checks_protocol_and_matrix_hashes(tmp_path):
    bundle_path = tmp_path / "input.zip"
    _write_input_bundle(bundle_path)

    protocol, summary = _load_input_bundle(bundle_path)

    assert protocol["protocol_id"] == summary["protocol_id"]
    assert len(protocol["dataset"]["languages"]) == 10


def test_input_bundle_rejects_matrix_hash_mismatch(tmp_path):
    bundle_path = tmp_path / "input.zip"
    _write_input_bundle(bundle_path, bad_matrix_hash=True)

    with pytest.raises(ValueError, match="Prior-matrix checksum mismatch"):
        _load_input_bundle(bundle_path)


def test_input_bundle_rejects_previous_roster_protocol(tmp_path):
    bundle_path = tmp_path / "input.zip"
    _write_input_bundle(bundle_path, stale_protocol=True)

    with pytest.raises(ValueError, match="current frozen RQ1 v2 protocol"):
        _load_input_bundle(bundle_path)


def test_incomplete_published_interference_matrix_is_rejected(tmp_path):
    matrix_path = tmp_path / "interference.csv"
    matrix_path.write_text(
        "language,en,hi\n"
        "en,0,-0.1\n"
        "hi,-0.2,0\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="no score for"):
        _validated_interference_matrix(
            matrix_path,
            ["en", "hi", "es"],
        )


def test_imputed_interference_prior_is_written_in_rq1_matrix_codes(
    tmp_path,
    monkeypatch,
):
    languages = phase4_interference_matrix.PHASE4_LANGUAGES
    scores = {
        past: {
            incoming: 0.0 if past == incoming else -0.1
            for incoming in languages
        }
        for past in languages
    }
    metadata = {
        "source_matrix_sha256": "a" * 64,
        "imputation": {"language": "ar", "k": 5},
    }
    monkeypatch.setattr(
        phase4_interference_matrix,
        "prepare_phase4_matrix",
        lambda source: (scores, metadata),
    )
    output = tmp_path / "priors" / "interference_matrix.csv"

    written_metadata = _write_imputed_interference_prior(
        output,
        tmp_path / "source.csv",
    )

    matrix_codes = {
        language: LANGUAGE_METADATA[language]["matrix_code"]
        for language in languages
    }
    assert set(load_wide_matrix(output)) == set(matrix_codes.values())
    assert written_metadata["imputation"]["language"] == "ar"
    assert written_metadata["matrix_sha256"]
