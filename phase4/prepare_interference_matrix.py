"""Prepare the ten-language Phase 4 interference matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase4.interference_matrix import (
    DEFAULT_SOURCE_MATRIX,
    PROJECT_ROOT,
    prepare_phase4_matrix,
    write_phase4_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, default=DEFAULT_SOURCE_MATRIX)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "phase4" / "artifacts" / "interference_matrix_10lang.csv",
    )
    args = parser.parse_args()

    scores, metadata = prepare_phase4_matrix(args.source_matrix)
    metadata = write_phase4_matrix(scores, metadata, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
