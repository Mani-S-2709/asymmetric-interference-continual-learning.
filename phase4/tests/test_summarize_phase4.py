import json

from phase2.prepare_data import hash_data_artifacts
from phase2.protocol import build_manifest, load_protocol
from phase4.run_phase4 import DEFAULT_PROTOCOL
from phase4.summarize_phase4 import aggregate_phase4


def test_aggregate_phase4_uses_saved_manifest_without_raw_dataset_files(tmp_path):
    protocol = load_protocol(DEFAULT_PROTOCOL)
    manifest = build_manifest(protocol)
    data_artifacts = {
        language: {
            split: {
                "path": f"{language}.{split}.jsonl",
                "rows": protocol["dataset"][f"{split}_examples_per_language"],
                "dataset_size": 1000,
                "sha256": "0" * 64,
            }
            for split in ("train", "test")
        }
        for language in protocol["dataset"]["languages"]
    }
    saved_manifest = {
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "data_artifacts": data_artifacts,
        "data_manifest_sha256": hash_data_artifacts(data_artifacts),
    }
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps(saved_manifest),
        encoding="utf-8",
    )

    report = aggregate_phase4(DEFAULT_PROTOCOL, tmp_path)

    assert report["expected_run_count"] == 15
    assert report["completed_run_count"] == 0
    assert len(report["missing_runs"]) == 15
