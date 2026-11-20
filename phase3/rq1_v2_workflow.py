"""Prepare, run, and analyze the Phase 3 ten-language RQ1 workflow."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from phase2.prepare_data import prepare_xnli
from phase2.preflight import collect_preflight
from phase2.protocol import build_manifest, load_protocol, validate_protocol
from phase2.results import validate_result_record
from phase2.run_baseline import validate_prepared_artifacts
from phase3.prepare_embedding_similarity import prepare as prepare_embedding
from phase3.prepare_grambank_distance import prepare as prepare_grambank
from phase3.prepare_urielplus_distance import (
    generate_matrix as generate_uriel_matrix,
    write_matrix as write_uriel_matrix,
)
from phase3.prior_analysis import (
    LANGUAGE_METADATA,
    PriorMatrix,
    load_wide_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    Path(__file__).parent / "configs" / "rq1_v2_10lang_protocol.json"
)
INPUT_ZIP_NAME = "phase3_rq1_v2_inputs.zip"
INPUT_SUMMARY_NAME = "rq1_v2_input_summary.json"
PROTOCOL_ARCHIVE_NAME = "protocol.json"
SEED_SUMMARY_NAME = "rq1_v2_b0_seed_summary.json"
ORDER_SUMMARY_NAME = "rq1_v2_b0_order_summary.json"
SEED_MANIFEST_NAME = "protocol_manifest.json"
ANALYSIS_ZIP_NAME = "phase3_rq1_v2_analysis.zip"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_archive_json(archive: ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name))
    except KeyError as exc:
        raise ValueError(f"Archive is missing {name!r}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Archive member {name!r} must contain a JSON object.")
    return value


def _load_input_bundle(
    input_zip: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with ZipFile(input_zip) as archive:
        summary = _read_archive_json(archive, INPUT_SUMMARY_NAME)
        protocol = _read_archive_json(archive, PROTOCOL_ARCHIVE_NAME)
    validate_protocol(protocol)
    manifest = build_manifest(protocol)
    current_protocol = load_protocol(DEFAULT_PROTOCOL)
    current_manifest = build_manifest(current_protocol)
    if (
        protocol.get("protocol_id") != current_protocol.get("protocol_id")
        or manifest["protocol_sha256"] != current_manifest["protocol_sha256"]
    ):
        raise ValueError(
            "Input bundle does not match the current frozen RQ1 v2 protocol. "
            "Rerun input preparation before training or analysis."
        )
    if summary.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("Input bundle protocol ID does not match protocol.json.")
    if summary.get("protocol_sha256") != manifest["protocol_sha256"]:
        raise ValueError("Input bundle protocol hash does not match protocol.json.")
    data_hash = summary.get("data_manifest_sha256")
    if not isinstance(data_hash, str) or len(data_hash) != 64:
        raise ValueError("Input bundle is missing a valid data-manifest SHA-256.")
    if summary.get("languages") != protocol["dataset"]["languages"]:
        raise ValueError("Input bundle language list does not match protocol.json.")
    if summary.get("orders") != [order["id"] for order in protocol["orders"]]:
        raise ValueError("Input bundle order list does not match protocol.json.")
    if summary.get("seeds") != protocol["training"]["seeds"]:
        raise ValueError("Input bundle seed list does not match protocol.json.")
    hashes = summary.get("prior_matrix_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Input bundle is missing prior-matrix checksums.")
    if "published_interference" in summary.get("included_priors", []):
        interference_metadata = summary.get("interference_metadata")
        matrix_hash = hashes.get("priors/interference_matrix.csv")
        if (
            not isinstance(interference_metadata, dict)
            or not isinstance(matrix_hash, str)
            or interference_metadata.get("matrix_sha256") != matrix_hash
        ):
            raise ValueError(
                "Input bundle is missing matching published-interference provenance."
            )
        imputation = interference_metadata.get("imputation")
        if imputation is not None:
            if not isinstance(imputation, dict) or (
                imputation.get("method") != "uniform_mean_of_k_nearest_languages"
                or imputation.get("language") != "ar"
                or imputation.get("k") != 5
                or imputation.get("distance") != "URIEL+ syntactic"
            ):
                raise ValueError(
                    "Input bundle has unexpected Arabic-interference imputation metadata."
                )
    with ZipFile(input_zip) as archive:
        for name, expected_hash in hashes.items():
            if not isinstance(name, str) or not name.startswith("priors/"):
                raise ValueError(f"Invalid prior-matrix archive path: {name!r}.")
            try:
                payload = archive.read(name)
            except KeyError as exc:
                raise ValueError(f"Input bundle is missing {name!r}.") from exc
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Prior-matrix checksum mismatch for {name}.")
    return protocol, summary


def _validated_interference_matrix(
    path: str | Path,
    languages: list[str],
) -> dict[str, dict[str, float]]:
    values = load_wide_matrix(path)
    prior = PriorMatrix(
        name="published_interference",
        values=values,
        value_kind="paper_interference",
        language_codes={
            language: LANGUAGE_METADATA[language]["matrix_code"]
            for language in languages
        },
    )
    for past_language in languages:
        for incoming_language in languages:
            if past_language != incoming_language:
                prior.score(past_language, incoming_language)
    return values


def _validate_prior_matrix(
    path: str | Path,
    languages: list[str],
    *,
    name: str,
    value_kind: str,
) -> None:
    prior = PriorMatrix(
        name=name,
        values=load_wide_matrix(path),
        value_kind=value_kind,
        language_codes={
            language: LANGUAGE_METADATA[language]["matrix_code"]
            for language in languages
        },
    )
    for past_language in languages:
        for incoming_language in languages:
            prior.score(past_language, incoming_language)


def _write_imputed_interference_prior(
    output_path: str | Path,
    source_matrix: str | Path | None = None,
) -> dict[str, Any]:
    """Write the frozen Phase 4 Arabic-imputed matrix in RQ1 matrix codes."""
    from phase4.interference_matrix import (
        DEFAULT_SOURCE_MATRIX,
        PHASE4_LANGUAGES,
        prepare_phase4_matrix,
    )

    languages = list(PHASE4_LANGUAGES)
    if languages != [
        "en", "de", "es", "fr", "sw", "ar", "ur", "bg", "ru", "hi"
    ]:
        raise ValueError(
            "Phase 4 language roster does not match the frozen RQ1 v2 roster."
        )
    scores, metadata = prepare_phase4_matrix(
        source_matrix or DEFAULT_SOURCE_MATRIX,
    )
    matrix_codes = {
        language: LANGUAGE_METADATA[language]["matrix_code"]
        for language in languages
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "language",
            *(matrix_codes[language] for language in languages),
        ])
        for past_language in languages:
            writer.writerow([
                matrix_codes[past_language],
                *(scores[past_language][incoming] for incoming in languages),
            ])
    _validate_prior_matrix(
        output,
        languages,
        name="published_interference",
        value_kind="paper_interference",
    )
    result = dict(metadata)
    result["matrix_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    result["matrix_path"] = "priors/interference_matrix.csv"
    result["matrix_codes"] = matrix_codes
    return result


def prepare_inputs(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_dir: str | Path,
    embedding_device: str | None = None,
    interference_matrix: str | Path | None = None,
    impute_arabic_interference: bool = False,
    interference_source_matrix: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare fixed XNLI-derived priors and package their provenance."""
    if interference_matrix is not None and impute_arabic_interference:
        raise ValueError(
            "Supply either a complete interference matrix or request the "
            "frozen Arabic imputation, not both."
        )
    if interference_source_matrix is not None and not impute_arabic_interference:
        raise ValueError(
            "An interference source matrix requires Arabic imputation."
        )
    protocol = load_protocol(protocol_path)
    languages = protocol["dataset"]["languages"]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    priors_dir = output_root / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir = output_root / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="rq1-v2-data-", dir=output_root
    ) as temporary:
        prepared_data_dir = Path(temporary) / "prepared_data"
        data_manifest = prepare_xnli(protocol_path, prepared_data_dir)

        uriel_codes, uriel_values, uriel_version = generate_uriel_matrix(languages)
        uriel_path = priors_dir / "uriel_syntactic_distance.csv"
        uriel_metadata = write_uriel_matrix(
            uriel_path,
            uriel_codes,
            uriel_values,
            uriel_version,
        )
        _validate_prior_matrix(
            uriel_path,
            languages,
            name="uriel_syntactic_distance",
            value_kind="distance",
        )

        embedding_path = priors_dir / "embedding_similarity.csv"
        embedding_metadata = prepare_embedding(
            protocol_path,
            prepared_data_dir,
            embedding_path,
            device=embedding_device,
        )
        _validate_prior_matrix(
            embedding_path,
            languages,
            name="embedding_similarity",
            value_kind="similarity",
        )

        grambank_dir = output_root / "grambank"
        grambank_exclusion = None
        try:
            grambank_metadata = prepare_grambank(
                grambank_dir,
                languages=languages,
            )
            grambank_path = grambank_dir / "grambank_distance.csv"
            shutil.copy2(grambank_path, priors_dir / grambank_path.name)
            shutil.copy2(
                grambank_dir / "grambank_distance_metadata.json",
                priors_dir / "grambank_distance_metadata.json",
            )
        except ValueError as exc:
            if not str(exc).startswith(
                "The pinned Grambank release does not cover every requested language:"
            ):
                raise
            coverage_path = grambank_dir / "grambank_coverage.json"
            if not coverage_path.is_file():
                raise
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            missing = coverage.get("missing_project_languages")
            if coverage.get("matrix_generated") is not False or not missing:
                raise
            grambank_metadata = None
            grambank_exclusion = (
                "Grambank v1.0.3 does not cover every configured language: "
                + ", ".join(missing)
                + ". No values were imputed."
            )
            shutil.copy2(
                coverage_path,
                coverage_dir / "grambank_coverage.json",
            )
        coverage_path = grambank_dir / "grambank_coverage.json"
        if coverage_path.is_file():
            shutil.copy2(
                coverage_path,
                coverage_dir / "grambank_coverage.json",
            )

        included_priors = [
            "script_mismatch",
            "family_mismatch",
            "uriel_syntactic_distance",
            "embedding_similarity",
            "composite_script_uriel",
        ]
        excluded_priors: dict[str, str] = {}
        if grambank_exclusion:
            excluded_priors["grambank_distance"] = grambank_exclusion
        elif grambank_metadata is not None:
            included_priors.append("grambank_distance")

        interference_metadata = None
        if interference_matrix is not None:
            _validated_interference_matrix(interference_matrix, languages)
            interference_path = priors_dir / "interference_matrix.csv"
            shutil.copy2(interference_matrix, interference_path)
            included_priors.extend([
                "published_interference",
                "composite_interference_script",
            ])
            interference_metadata = {
                "matrix_sha256": hashlib.sha256(
                    interference_path.read_bytes()
                ).hexdigest(),
                "matrix_path": "priors/interference_matrix.csv",
                "source": "complete supplied 10-language matrix",
            }
        elif impute_arabic_interference:
            interference_path = priors_dir / "interference_matrix.csv"
            interference_metadata = _write_imputed_interference_prior(
                interference_path,
                interference_source_matrix,
            )
            _write_json(
                priors_dir / "interference_matrix_metadata.json",
                interference_metadata,
            )
            included_priors.extend([
                "published_interference",
                "composite_interference_script",
            ])
        else:
            excluded_priors["published_interference"] = (
                "No complete 10-language published-interference matrix was "
                "supplied and Arabic imputation was not requested."
            )

    matrix_files = sorted(priors_dir.glob("*.csv"))
    prior_hashes = {
        f"priors/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in matrix_files
    }
    if "grambank_coverage.json" in {
        path.name for path in coverage_dir.iterdir()
    }:
        shutil.copy2(
            coverage_dir / "grambank_coverage.json",
            output_root / "grambank_coverage.json",
        )

    protocol_manifest = build_manifest(protocol)
    summary = {
        "workflow": "phase3_rq1_v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_manifest["protocol_sha256"],
        "data_manifest_sha256": data_manifest["data_manifest_sha256"],
        "languages": languages,
        "seeds": protocol["training"]["seeds"],
        "orders": [order["id"] for order in protocol["orders"]],
        "expected_b0_runs": (
            len(protocol["training"]["seeds"]) * len(protocol["orders"])
        ),
        "included_priors": included_priors,
        "excluded_priors": excluded_priors,
        "prior_matrix_sha256": prior_hashes,
        "uriel_metadata": uriel_metadata,
        "embedding_metadata": embedding_metadata,
        "interference_metadata": interference_metadata,
        "grambank_metadata": grambank_metadata,
    }
    _write_json(output_root / INPUT_SUMMARY_NAME, summary)
    shutil.copy2(protocol_path, output_root / PROTOCOL_ARCHIVE_NAME)

    zip_path = output_root / INPUT_ZIP_NAME
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(output_root / INPUT_SUMMARY_NAME, INPUT_SUMMARY_NAME)
        archive.write(
            output_root / PROTOCOL_ARCHIVE_NAME,
            PROTOCOL_ARCHIVE_NAME,
        )
        for path in sorted(priors_dir.glob("*")):
            if path.is_file():
                archive.write(path, f"priors/{path.name}")
        coverage_path = output_root / "grambank_coverage.json"
        if coverage_path.is_file():
            archive.write(coverage_path, "grambank_coverage.json")
    summary["input_zip"] = str(zip_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Input bundle: {zip_path}")
    return summary


def train_seed(
    *,
    input_zip: str | Path,
    seed: int,
    work_dir: str | Path,
    output_dir: str | Path,
    order_id: str | None = None,
) -> dict[str, Any]:
    """Run B0 for one seed, optionally limiting execution to one order."""
    base_protocol, input_summary = _load_input_bundle(input_zip)
    if seed not in base_protocol["training"]["seeds"]:
        raise ValueError(
            f"Seed {seed} is not listed in the frozen protocol: "
            f"{base_protocol['training']['seeds']}."
        )
    protocol = copy.deepcopy(base_protocol)
    protocol["protocol_id"] = f"{base_protocol['protocol_id']}-seed-{seed}"
    protocol["training"]["seeds"] = [seed]
    validate_protocol(protocol)
    all_order_ids = [order["id"] for order in protocol["orders"]]
    if order_id is not None and order_id not in all_order_ids:
        raise ValueError(
            f"Unknown order {order_id!r}; expected one of {all_order_ids}."
        )
    order_ids = [order_id] if order_id is not None else all_order_ids

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_dir)
    if work_root.exists():
        raise FileExistsError(
            f"Work directory already exists: {work_root}. "
            "Use a fresh Kaggle /kaggle/temp directory for this seed."
        )
    work_root.mkdir(parents=True)
    protocol_path = work_root / "protocol.json"
    _write_json(protocol_path, protocol)

    preflight = collect_preflight(
        require_cuda=True,
        check_gated_model=True,
        protocol_path=protocol_path,
    )
    print(json.dumps(preflight, indent=2, sort_keys=True))

    artifact_root = work_root / "artifacts"
    prepared = prepare_xnli(protocol_path, artifact_root)
    validated_manifest = validate_prepared_artifacts(protocol, artifact_root)
    actual_data_hash = validated_manifest["data_manifest_sha256"]
    if actual_data_hash != input_summary["data_manifest_sha256"]:
        raise ValueError(
            "Prepared XNLI sample hash differs from the frozen input bundle; "
            "training did not start."
        )

    command = [
        sys.executable,
        "-m",
        "phase2.run_baseline",
        "--method",
        "B0",
        "--protocol",
        str(protocol_path),
        "--artifact-root",
        str(artifact_root),
    ]
    for selected_order_id in order_ids:
        command.extend(["--order-id", selected_order_id])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    run_manifest = build_manifest(protocol)
    run_manifest["data_artifacts"] = prepared["data_artifacts"]
    run_manifest["data_manifest_sha256"] = actual_data_hash
    expected_runs = [
        run for run in run_manifest["runs"] if run["method_id"] == "B0"
    ]
    expected_runs = [
        run for run in expected_runs if run["order_id"] in order_ids
    ]
    if len(expected_runs) != len(order_ids):
        raise ValueError(
            f"Expected {len(order_ids)} B0 records for seed {seed}; "
            f"found {len(expected_runs)}."
        )

    export_suffix = f"order_{order_id}" if order_id is not None else "export"
    export_root = output_root / f"seed_{seed}_{export_suffix}"
    results_dir = export_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for run in expected_runs:
        result_path = artifact_root / "results" / f"{run['run_id']}.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Missing completed B0 result: {result_path}."
            )
        record = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result_record(record, run_manifest)
        records.append(record)
        shutil.copy2(result_path, results_dir / result_path.name)
    if {record["order_id"] for record in records} != set(order_ids):
        raise ValueError("Completed B0 records do not cover the requested order set.")

    metrics = ("ACC", "BWT", "FWT", "F")
    summary = {
        "workflow": "phase3_rq1_v2",
        "protocol_id": base_protocol["protocol_id"],
        "seed_protocol_id": protocol["protocol_id"],
        "seed": seed,
        "completed_runs": len(records),
        "expected_runs": len(order_ids),
        "orders": order_ids,
        "data_manifest_sha256": actual_data_hash,
        "protocol_sha256": run_manifest["protocol_sha256"],
        "metrics_mean_across_orders": {
            name: statistics.mean(
                record["metrics"][name] for record in records
            )
            for name in metrics
        },
        "metrics_population_std_across_orders": {
            name: statistics.pstdev(
                record["metrics"][name] for record in records
            )
            for name in metrics
        },
    }
    if order_id is not None:
        summary["expected_orders"] = all_order_ids
        summary["order_id"] = order_id
        summary_name = ORDER_SUMMARY_NAME
        zip_name = f"phase3_rq1_v2_b0_seed_{seed}_{order_id}.zip"
    else:
        summary_name = SEED_SUMMARY_NAME
        zip_name = f"phase3_rq1_v2_b0_seed_{seed}.zip"
    _write_json(export_root / summary_name, summary)
    _write_json(export_root / SEED_MANIFEST_NAME, run_manifest)
    shutil.copy2(protocol_path, export_root / "protocol.json")

    zip_path = output_root / zip_name
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(export_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(export_root).as_posix(),
                )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Result bundle: {zip_path}")
    return summary


def analyze(
    *,
    input_zip: str | Path,
    result_zips: list[str | Path],
    output_dir: str | Path,
    random_repetitions: int = 10000,
) -> dict[str, Any]:
    """Validate complete-seed or per-order archives and run combined analysis."""
    protocol, input_summary = _load_input_bundle(input_zip)
    expected_seeds = set(protocol["training"]["seeds"])
    ordered_orders = [order["id"] for order in protocol["orders"]]
    expected_orders = set(ordered_orders)
    seed_archive_count = len(expected_seeds)
    order_archive_count = seed_archive_count * len(expected_orders)
    if len(result_zips) not in (seed_archive_count, order_archive_count):
        raise ValueError(
            f"Expected either {seed_archive_count} complete-seed archives or "
            f"{order_archive_count} per-order archives; received {len(result_zips)}."
        )
    per_order_archives = len(result_zips) == order_archive_count

    output_root = Path(output_dir)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries = []
    seen_seeds: set[int] = set()
    seen_seed_orders: set[tuple[int, str]] = set()
    order_records_by_seed: dict[int, list[dict[str, Any]]] = {}
    run_ids: set[str] = set()
    for result_zip in result_zips:
        with ZipFile(result_zip) as archive:
            run_manifest = _read_archive_json(archive, SEED_MANIFEST_NAME)
            if per_order_archives:
                order_summary = _read_archive_json(archive, ORDER_SUMMARY_NAME)
                seed = order_summary.get("seed")
                order_id = order_summary.get("order_id")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(f"Invalid seed in {result_zip}.")
                if seed not in expected_seeds:
                    raise ValueError(f"Unexpected training seed {seed}.")
                if not isinstance(order_id, str) or order_id not in expected_orders:
                    raise ValueError(f"Unexpected order in {result_zip}.")
                if (seed, order_id) in seen_seed_orders:
                    raise ValueError(
                        f"Duplicate result archive for seed {seed}, order {order_id}."
                    )
                if order_summary.get("protocol_id") != protocol["protocol_id"]:
                    raise ValueError(f"Protocol mismatch in {result_zip}.")
                if order_summary.get("seed_protocol_id") != (
                    f"{protocol['protocol_id']}-seed-{seed}"
                ):
                    raise ValueError(f"Seed protocol mismatch in {result_zip}.")
                if order_summary.get("data_manifest_sha256") != input_summary[
                    "data_manifest_sha256"
                ]:
                    raise ValueError(f"XNLI sample hash mismatch in {result_zip}.")
                if run_manifest.get("data_manifest_sha256") != input_summary[
                    "data_manifest_sha256"
                ]:
                    raise ValueError(
                        f"Result manifest data hash mismatch in {result_zip}."
                    )
                if order_summary.get("protocol_sha256") != run_manifest.get(
                    "protocol_sha256"
                ):
                    raise ValueError(
                        f"Result protocol hash mismatch in {result_zip}."
                    )
                if order_summary.get("completed_runs") != 1 or (
                    order_summary.get("orders") != [order_id]
                    or order_summary.get("expected_orders") != ordered_orders
                ):
                    raise ValueError(f"Incomplete order summary in {result_zip}.")
                records = [
                    name for name in archive.namelist()
                    if name.startswith("results/") and name.endswith(".json")
                ]
                if len(records) != 1:
                    raise ValueError(
                        f"Expected one result JSON in {result_zip}; "
                        f"found {len(records)}."
                    )
                name = records[0]
                record = json.loads(archive.read(name))
                validate_result_record(record, run_manifest)
                if record.get("training_seed") != seed:
                    raise ValueError(
                        f"Result {name} has a different training seed."
                    )
                if record.get("order_id") != order_id:
                    raise ValueError(
                        f"Result {name} does not match its order summary."
                    )
                if record["run_id"] in run_ids:
                    raise ValueError(f"Duplicate run ID: {record['run_id']}.")
                run_ids.add(record["run_id"])
                seen_seed_orders.add((seed, order_id))
                order_records_by_seed.setdefault(seed, []).append(record)
                seen_seeds.add(seed)
                destination = results_dir / Path(name).name
                destination.write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                continue

            seed_summary = _read_archive_json(archive, SEED_SUMMARY_NAME)
            seed = seed_summary.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"Invalid seed in {result_zip}.")
            if seed not in expected_seeds or seed in seen_seeds:
                raise ValueError(
                    f"Unexpected or duplicate training seed {seed}."
                )
            if seed_summary.get("protocol_id") != protocol["protocol_id"]:
                raise ValueError(f"Protocol mismatch in {result_zip}.")
            if (
                seed_summary.get("data_manifest_sha256")
                != input_summary["data_manifest_sha256"]
            ):
                raise ValueError(
                    f"XNLI sample hash mismatch in {result_zip}."
                )
            if run_manifest.get("data_manifest_sha256") != input_summary[
                "data_manifest_sha256"
            ]:
                raise ValueError(
                    f"Result manifest data hash mismatch in {result_zip}."
                )
            records = [
                name for name in archive.namelist()
                if name.startswith("results/") and name.endswith(".json")
            ]
            if len(records) != len(expected_orders):
                raise ValueError(
                    f"Expected five result JSONs in {result_zip}; "
                    f"found {len(records)}."
                )
            found_orders = set()
            for name in records:
                record = json.loads(archive.read(name))
                validate_result_record(record, run_manifest)
                if record.get("training_seed") != seed:
                    raise ValueError(
                        f"Result {name} has a different training seed."
                    )
                if record.get("order_id") not in expected_orders:
                    raise ValueError(f"Unexpected order in {name}.")
                if record["run_id"] in run_ids:
                    raise ValueError(f"Duplicate run ID: {record['run_id']}.")
                run_ids.add(record["run_id"])
                found_orders.add(record["order_id"])
                destination = results_dir / Path(name).name
                destination.write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if found_orders != expected_orders:
                raise ValueError(
                    f"Result archive for seed {seed} does not cover all orders."
                )
            if seed_summary.get("completed_runs") != len(expected_orders):
                raise ValueError(f"Incomplete run count in {result_zip}.")
            seed_summaries.append(seed_summary)
            seen_seeds.add(seed)

    if per_order_archives:
        expected_seed_orders = {
            (seed, order_id)
            for seed in expected_seeds
            for order_id in expected_orders
        }
        if seen_seed_orders != expected_seed_orders:
            missing = sorted(expected_seed_orders - seen_seed_orders)
            raise ValueError(f"Missing seed/order archives: {missing}.")
        metrics = ("ACC", "BWT", "FWT", "F")
        for seed in sorted(expected_seeds):
            records = order_records_by_seed[seed]
            records_by_order = {
                record["order_id"]: record for record in records
            }
            records = [records_by_order[order_id] for order_id in ordered_orders]
            seed_summaries.append({
                "workflow": "phase3_rq1_v2",
                "protocol_id": protocol["protocol_id"],
                "seed_protocol_id": f"{protocol['protocol_id']}-seed-{seed}",
                "seed": seed,
                "completed_runs": len(records),
                "expected_runs": len(ordered_orders),
                "orders": ordered_orders,
                "data_manifest_sha256": input_summary["data_manifest_sha256"],
                "metrics_mean_across_orders": {
                    name: statistics.mean(
                        record["metrics"][name] for record in records
                    )
                    for name in metrics
                },
                "metrics_population_std_across_orders": {
                    name: statistics.pstdev(
                        record["metrics"][name] for record in records
                    )
                    for name in metrics
                },
            })
    elif seen_seeds != expected_seeds:
        raise ValueError(
            f"Missing seed archives: {sorted(expected_seeds - seen_seeds)}."
        )

    priors_dir = output_root / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(input_zip) as archive:
        for name in archive.namelist():
            if name.startswith("priors/") and not name.endswith("/"):
                destination = priors_dir / Path(name).name
                destination.write_bytes(archive.read(name))

    analysis_dir = output_root / "analysis"
    command = [
        sys.executable,
        "-m",
        "phase3.analyze_priors",
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(analysis_dir),
        "--random-repetitions",
        str(random_repetitions),
    ]
    prior_paths = {
        "uriel_syntactic_distance": priors_dir / "uriel_syntactic_distance.csv",
        "grambank_distance": priors_dir / "grambank_distance.csv",
        "embedding_similarity": priors_dir / "embedding_similarity.csv",
        "published_interference": priors_dir / "interference_matrix.csv",
    }
    for prior_name, option in (
        ("uriel_syntactic_distance", "--uriel-distance"),
        ("grambank_distance", "--grambank-distance"),
        ("embedding_similarity", "--embedding-similarity"),
        ("published_interference", "--interference-matrix"),
    ):
        path = prior_paths[prior_name]
        if path.is_file():
            command.extend([option, str(path)])
    if not prior_paths["grambank_distance"].is_file():
        command.extend(["--exclude-prior", "grambank_distance"])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    provenance = {
        "workflow": "phase3_rq1_v2",
        "protocol_id": protocol["protocol_id"],
        "data_manifest_sha256": input_summary["data_manifest_sha256"],
        "included_priors": input_summary["included_priors"],
        "excluded_priors": input_summary["excluded_priors"],
        "interference_metadata": input_summary.get("interference_metadata"),
        "seed_summaries": sorted(seed_summaries, key=lambda item: item["seed"]),
        "input_bundle": Path(input_zip).name,
        "result_bundles": [Path(path).name for path in result_zips],
    }
    _write_json(output_root / "rq1_v2_analysis_provenance.json", provenance)
    analysis_zip = output_root / ANALYSIS_ZIP_NAME
    with ZipFile(analysis_zip, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(analysis_dir.glob("*")):
            if path.is_file():
                archive.write(path, f"analysis/{path.name}")
        archive.write(
            output_root / "rq1_v2_analysis_provenance.json",
            "rq1_v2_analysis_provenance.json",
        )
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Analysis bundle: {analysis_zip}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare-inputs",
        help="Prepare fixed XNLI-derived prior matrices and input ZIP.",
    )
    prepare_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--embedding-device",
        choices=("cpu", "cuda"),
        help="Embedding inference device; this is not model training.",
    )
    prepare_parser.add_argument("--interference-matrix", type=Path)
    prepare_parser.add_argument(
        "--impute-arabic-interference",
        action="store_true",
        help="Use the frozen Phase 4 k=5 URIEL+ rule for the missing Arabic scores.",
    )
    prepare_parser.add_argument(
        "--interference-source-matrix",
        type=Path,
        help="Published source matrix for Arabic imputation; defaults to Phase 4's source.",
    )

    train_parser = commands.add_parser(
        "train-seed",
        help="Run B0 for all five orders for one training seed.",
    )
    train_parser.add_argument("--inputs-zip", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--work-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)

    order_parser = commands.add_parser(
        "train-order",
        help="Run B0 for one order of one training seed.",
    )
    order_parser.add_argument("--inputs-zip", type=Path, required=True)
    order_parser.add_argument("--seed", type=int, required=True)
    order_parser.add_argument("--order-id", required=True)
    order_parser.add_argument("--work-dir", type=Path, required=True)
    order_parser.add_argument("--output-dir", type=Path, required=True)

    analyze_parser = commands.add_parser(
        "analyze",
        help="Validate seed result ZIPs and run combined RQ1 analysis.",
    )
    analyze_parser.add_argument("--inputs-zip", type=Path, required=True)
    analyze_parser.add_argument(
        "--result-zip",
        type=Path,
        action="append",
        required=True,
        help="Seed result ZIP; specify once per configured seed.",
    )
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--random-repetitions", type=int, default=10000)

    args = parser.parse_args()
    if args.command == "prepare-inputs":
        prepare_inputs(
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            embedding_device=args.embedding_device,
            interference_matrix=args.interference_matrix,
            impute_arabic_interference=args.impute_arabic_interference,
            interference_source_matrix=args.interference_source_matrix,
        )
    elif args.command == "train-seed":
        train_seed(
            input_zip=args.inputs_zip,
            seed=args.seed,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "train-order":
        train_seed(
            input_zip=args.inputs_zip,
            seed=args.seed,
            order_id=args.order_id,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
        )
    else:
        if args.random_repetitions < 1:
            parser.error("--random-repetitions must be positive.")
        analyze(
            input_zip=args.inputs_zip,
            result_zips=args.result_zip,
            output_dir=args.output_dir,
            random_repetitions=args.random_repetitions,
        )


if __name__ == "__main__":
    main()
