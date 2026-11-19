"""Aggregate completed Phase 4 pilot runs and list missing comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Any

from phase2.prepare_data import hash_data_artifacts
from phase2.protocol import build_manifest, load_protocol
from phase2.results import aggregate_results, write_aggregate
from phase2.run_baseline import validate_prepared_artifacts
from phase4.run_phase4 import DEFAULT_ARTIFACT_ROOT, DEFAULT_PROTOCOL


def _validate_saved_manifest(
    protocol: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    saved_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if saved_manifest.get("protocol_id") != expected_manifest["protocol_id"]:
        raise ValueError("Saved protocol manifest belongs to a different protocol.")
    if saved_manifest.get("protocol_sha256") != expected_manifest["protocol_sha256"]:
        raise ValueError("Saved protocol manifest does not match the frozen protocol.")
    artifacts = saved_manifest.get("data_artifacts")
    languages = protocol["dataset"]["languages"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(languages):
        raise ValueError("Saved data manifest has unexpected language coverage.")

    for language in languages:
        language_artifacts = artifacts[language]
        if not isinstance(language_artifacts, Mapping):
            raise ValueError(f"Invalid saved artifacts for {language!r}.")
        for split in ("train", "test"):
            artifact = language_artifacts.get(split)
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Missing saved {split} artifact for {language!r}.")
            filename = artifact.get("path")
            rows = artifact.get("rows")
            dataset_size = artifact.get("dataset_size")
            digest = artifact.get("sha256")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or rows != protocol["dataset"][f"{split}_examples_per_language"]
                or isinstance(dataset_size, bool)
                or not isinstance(dataset_size, int)
                or dataset_size < rows
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(
                    f"Invalid saved {split} artifact metadata for {language!r}."
                )
    if saved_manifest.get("data_manifest_sha256") != hash_data_artifacts(artifacts):
        raise ValueError("Saved data manifest metadata is missing or invalid.")
    return {
        "data_artifacts": artifacts,
        "data_manifest_sha256": saved_manifest["data_manifest_sha256"],
    }


def aggregate_phase4(
    protocol_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    manifest = build_manifest(protocol)
    root = Path(artifact_root)
    if (root / "data").is_dir():
        prepared = validate_prepared_artifacts(protocol, root)
    else:
        manifest_path = root / "protocol_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Protocol manifest not found: {manifest_path}.")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prepared = _validate_saved_manifest(protocol, manifest, saved_manifest)
    manifest["data_artifacts"] = prepared["data_artifacts"]
    manifest["data_manifest_sha256"] = prepared["data_manifest_sha256"]
    return aggregate_results(manifest, root / "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to <artifact-root>/aggregate.json.",
    )
    args = parser.parse_args()

    report = aggregate_phase4(args.protocol, args.artifact_root)
    output = args.output or Path(args.artifact_root) / "aggregate.json"
    write_aggregate(report, output)
    print(
        f"Wrote {output}; completed "
        f"{report['completed_run_count']}/{report['expected_run_count']} runs."
    )


if __name__ == "__main__":
    main()
