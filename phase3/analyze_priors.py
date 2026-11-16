"""Command-line entry point for the Phase 3 RQ1 prior comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.prior_analysis import (
    LANGUAGE_METADATA,
    PriorMatrix,
    compare_priors,
    compare_priors_by_order,
    load_sequential_forgetting,
    load_wide_matrix,
    make_composite_prior,
    make_builtin_prior_matrices,
    random_prior_control,
    write_analysis_outputs,
)


def _add_prior(
    priors: dict[str, PriorMatrix],
    path: str | None,
    *,
    name: str,
    value_kind: str,
) -> None:
    if path:
        priors[name] = PriorMatrix(
            name=name,
            values=load_wide_matrix(path),
            value_kind=value_kind,
            language_codes={
                language: metadata["matrix_code"]
                for language, metadata in LANGUAGE_METADATA.items()
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prior risk scores with per-language forgetting from "
            "completed B0 sequential result records."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "phase2" / "runs" / "b0_kaggle_2026-11-11" / "results",
        help="Directory containing completed B0 result JSON records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "phase3" / "results" / "rq1_pilot",
        help="Directory for CSV and JSON analysis outputs.",
    )
    parser.add_argument(
        "--interference-matrix",
        help="Wide CSV from the paper; rows=affected/past, columns=added/incoming.",
    )
    parser.add_argument("--uriel-distance", help="Optional wide CSV of URIEL+ distances.")
    parser.add_argument("--grambank-distance", help="Optional wide CSV of Grambank distances.")
    parser.add_argument(
        "--embedding-similarity",
        help="Optional wide CSV of symmetric embedding similarities.",
    )
    parser.add_argument(
        "--exclude-prior",
        choices=("grambank_distance",),
        action="append",
        default=[],
        help=(
            "Intentionally omit a prior from this analysis and record the "
            "reason in summary.json."
        ),
    )
    parser.add_argument("--random-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.grambank_distance and "grambank_distance" in args.exclude_prior:
        parser.error(
            "--grambank-distance cannot be supplied when that prior is excluded."
        )

    result_paths = sorted(args.results_dir.glob("*.json"))
    if not result_paths:
        parser.error(f"No result JSON files found in {args.results_dir}.")

    observations, pair_forgetting = load_sequential_forgetting(result_paths)
    languages = sorted({
        pair.past_language for pair in pair_forgetting
    } | {
        pair.incoming_language for pair in pair_forgetting
    })
    priors = make_builtin_prior_matrices(languages)
    missing_priors = []
    excluded_priors = {}
    _add_prior(
        priors, args.interference_matrix,
        name="published_interference", value_kind="paper_interference",
    )
    _add_prior(
        priors, args.uriel_distance,
        name="uriel_syntactic_distance", value_kind="distance",
    )
    _add_prior(
        priors, args.grambank_distance,
        name="grambank_distance", value_kind="distance",
    )
    _add_prior(
        priors, args.embedding_similarity,
        name="embedding_similarity", value_kind="similarity",
    )
    if not args.interference_matrix:
        missing_priors.append("published_interference")
    if not args.uriel_distance:
        missing_priors.append("uriel_syntactic_distance")
    if not args.grambank_distance:
        if "grambank_distance" in args.exclude_prior:
            excluded_priors["grambank_distance"] = (
                "Excluded for the five-language pilot: Grambank v1.0.3 lacks "
                "Spanish and German records."
            )
        else:
            missing_priors.append("grambank_distance")
    if not args.embedding_similarity:
        missing_priors.append("embedding_similarity")

    if "uriel_syntactic_distance" in priors:
        priors["composite_script_uriel"] = make_composite_prior(
            "composite_script_uriel",
            [priors["script_mismatch"], priors["uriel_syntactic_distance"]],
            languages,
        )
    else:
        missing_priors.append("composite_script_uriel")
    if "published_interference" in priors:
        priors["composite_interference_script"] = make_composite_prior(
            "composite_interference_script",
            [priors["published_interference"], priors["script_mismatch"]],
            languages,
        )
    else:
        missing_priors.append("composite_interference_script")

    correlations = compare_priors(pair_forgetting, priors)
    order_correlations = compare_priors_by_order(observations, priors)
    random_control = random_prior_control(
        pair_forgetting, repetitions=args.random_repetitions
    )
    write_analysis_outputs(
        args.output_dir,
        observations,
        pair_forgetting,
        correlations,
        order_correlations,
        random_control,
        missing_priors=missing_priors,
        excluded_priors=excluded_priors,
    )
    print(f"Analyzed {len(result_paths)} B0 runs and {len(pair_forgetting)} directed language pairs.")
    print(f"Results: {args.output_dir.resolve()}")
    if missing_priors:
        print("Missing prior inputs: " + ", ".join(missing_priors))
        print("The analysis is partial; do not treat it as the completed Phase 3 comparison.")


if __name__ == "__main__":
    main()
