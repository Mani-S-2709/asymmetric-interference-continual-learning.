import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from phase2.protocol import build_manifest, load_protocol
from phase2.results import make_sequential_record, write_result_record
import phase3.rq1_v2_workflow as workflow
from phase3.rq1_v2_workflow import (
    INPUT_SUMMARY_NAME,
    ORDER_SUMMARY_NAME,
    PROTOCOL_ARCHIVE_NAME,
    SEED_MANIFEST_NAME,
    SEED_SUMMARY_NAME,
    _load_input_bundle,
    _validated_interference_matrix,
    _write_imputed_interference_prior,
    analyze,
    train_seed,
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
        "included_priors": [],
        "excluded_priors": [],
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


def _make_record(manifest, run):
    task_count = len(run["task_order"])
    return make_sequential_record(
        manifest=manifest,
        run=run,
        performance_matrix=[
            [0.2 + 0.01 * row for _ in range(task_count)]
            for row in range(task_count)
        ],
        zero_shot_accuracy=[0.2] * task_count,
    )


def test_train_seed_can_export_a_single_order(tmp_path, monkeypatch):
    input_zip = tmp_path / "input.zip"
    _write_input_bundle(input_zip)
    monkeypatch.setattr(workflow, "collect_preflight", lambda **kwargs: {})

    def prepare_xnli(protocol_path, artifact_root):
        Path(artifact_root).mkdir(parents=True)
        return {"data_artifacts": {}}

    monkeypatch.setattr(workflow, "prepare_xnli", prepare_xnli)
    monkeypatch.setattr(
        workflow,
        "validate_prepared_artifacts",
        lambda protocol, artifact_root: {
            "data_manifest_sha256": "a" * 64
        },
    )

    def run_selected_order(command, cwd, check):
        protocol_path = Path(command[command.index("--protocol") + 1])
        artifact_root = Path(command[command.index("--artifact-root") + 1])
        order_ids = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--order-id"
        ]
        assert order_ids == ["order-1"]
        manifest = build_manifest(load_protocol(protocol_path))
        manifest["data_manifest_sha256"] = "a" * 64
        run = next(
            item for item in manifest["runs"]
            if item["method_id"] == "B0"
            and item["order_id"] == "order-1"
        )
        write_result_record(
            _make_record(manifest, run),
            artifact_root / "results",
            manifest,
        )

    monkeypatch.setattr(workflow.subprocess, "run", run_selected_order)
    output_dir = tmp_path / "output"
    summary = train_seed(
        input_zip=input_zip,
        seed=42,
        order_id="order-1",
        work_dir=tmp_path / "work",
        output_dir=output_dir,
    )

    result_zip = output_dir / "phase3_rq1_v2_b0_seed_42_order-1.zip"
    assert result_zip.is_file()
    assert summary["orders"] == ["order-1"]
    assert summary["completed_runs"] == 1
    with ZipFile(result_zip) as archive:
        order_summary = json.loads(archive.read(ORDER_SUMMARY_NAME))
        assert order_summary["expected_orders"] == [
            f"order-{index}" for index in range(1, 6)
        ]
        assert len([
            name for name in archive.namelist()
            if name.startswith("results/") and name.endswith(".json")
        ]) == 1


def test_analyze_accepts_all_per_order_archives(tmp_path, monkeypatch):
    input_zip = tmp_path / "input.zip"
    _write_input_bundle(input_zip)
    protocol = load_protocol(PROTOCOL_PATH)
    result_zips = []
    data_hash = "a" * 64

    for seed in protocol["training"]["seeds"]:
        seed_protocol = copy.deepcopy(protocol)
        seed_protocol["protocol_id"] = (
            f"{protocol['protocol_id']}-seed-{seed}"
        )
        seed_protocol["training"]["seeds"] = [seed]
        manifest = build_manifest(seed_protocol)
        manifest["data_manifest_sha256"] = data_hash
        for order_id in [order["id"] for order in protocol["orders"]]:
            run = next(
                item for item in manifest["runs"]
                if item["method_id"] == "B0"
                and item["order_id"] == order_id
            )
            result_zip = tmp_path / (
                f"phase3_rq1_v2_b0_seed_{seed}_{order_id}.zip"
            )
            order_summary = {
                "workflow": "phase3_rq1_v2",
                "protocol_id": protocol["protocol_id"],
                "seed_protocol_id": seed_protocol["protocol_id"],
                "seed": seed,
                "completed_runs": 1,
                "expected_runs": 1,
                "orders": [order_id],
                "expected_orders": [
                    order["id"] for order in protocol["orders"]
                ],
                "order_id": order_id,
                "data_manifest_sha256": data_hash,
                "protocol_sha256": manifest["protocol_sha256"],
            }
            with ZipFile(result_zip, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr(
                    ORDER_SUMMARY_NAME,
                    json.dumps(order_summary),
                )
                archive.writestr(
                    SEED_MANIFEST_NAME,
                    json.dumps(manifest),
                )
                archive.writestr(
                    f"results/{run['run_id']}.json",
                    json.dumps(_make_record(manifest, run)),
                )
            result_zips.append(result_zip)

    def run_analysis(command, cwd, check):
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        analysis_dir.mkdir(parents=True)
        (analysis_dir / "summary.json").write_text(
            json.dumps({"status": "ok"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(workflow.subprocess, "run", run_analysis)
    output_dir = tmp_path / "analysis-output"
    summary = analyze(
        input_zip=input_zip,
        result_zips=result_zips,
        output_dir=output_dir,
    )

    assert summary == {"status": "ok"}
    assert len(list((output_dir / "results").glob("*.json"))) == 15
    provenance = json.loads(
        (output_dir / "rq1_v2_analysis_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["completed_runs"] for item in provenance["seed_summaries"]] == [
        5, 5, 5
    ]


def test_analyze_still_accepts_complete_seed_archives(tmp_path, monkeypatch):
    input_zip = tmp_path / "input.zip"
    _write_input_bundle(input_zip)
    protocol = load_protocol(PROTOCOL_PATH)
    result_zips = []
    data_hash = "a" * 64

    for seed in protocol["training"]["seeds"]:
        seed_protocol = copy.deepcopy(protocol)
        seed_protocol["protocol_id"] = (
            f"{protocol['protocol_id']}-seed-{seed}"
        )
        seed_protocol["training"]["seeds"] = [seed]
        manifest = build_manifest(seed_protocol)
        manifest["data_manifest_sha256"] = data_hash
        b0_runs = [
            run for run in manifest["runs"] if run["method_id"] == "B0"
        ]
        result_zip = tmp_path / f"phase3_rq1_v2_b0_seed_{seed}.zip"
        seed_summary = {
            "workflow": "phase3_rq1_v2",
            "protocol_id": protocol["protocol_id"],
            "seed_protocol_id": seed_protocol["protocol_id"],
            "seed": seed,
            "completed_runs": len(b0_runs),
            "expected_runs": len(b0_runs),
            "orders": [order["id"] for order in protocol["orders"]],
            "data_manifest_sha256": data_hash,
        }
        with ZipFile(result_zip, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(SEED_SUMMARY_NAME, json.dumps(seed_summary))
            archive.writestr(SEED_MANIFEST_NAME, json.dumps(manifest))
            for run in b0_runs:
                archive.writestr(
                    f"results/{run['run_id']}.json",
                    json.dumps(_make_record(manifest, run)),
                )
        result_zips.append(result_zip)

    def run_analysis(command, cwd, check):
        analysis_dir = Path(command[command.index("--output-dir") + 1])
        analysis_dir.mkdir(parents=True)
        (analysis_dir / "summary.json").write_text(
            json.dumps({"status": "ok"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(workflow.subprocess, "run", run_analysis)
    output_dir = tmp_path / "analysis-output"
    summary = analyze(
        input_zip=input_zip,
        result_zips=result_zips,
        output_dir=output_dir,
    )

    assert summary == {"status": "ok"}
    assert len(list((output_dir / "results").glob("*.json"))) == 15


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
