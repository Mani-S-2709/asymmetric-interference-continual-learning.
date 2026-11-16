"""Analyze whether language priors predict forgetting in sequential runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from itertools import permutations
import json
import math
from pathlib import Path
import random
from typing import Any


LANGUAGE_METADATA: dict[str, dict[str, str]] = {
    "en": {
        "matrix_code": "eng_Latn", "iso3_code": "eng",
        "script": "Latn", "family": "Indo-European",
    },
    "hi": {
        "matrix_code": "hin_Deva", "iso3_code": "hin",
        "script": "Deva", "family": "Indo-European",
    },
    "es": {
        "matrix_code": "spa_Latn", "iso3_code": "spa",
        "script": "Latn", "family": "Indo-European",
    },
    "de": {
        "matrix_code": "deu_Latn", "iso3_code": "deu",
        "script": "Latn", "family": "Indo-European",
    },
    "sw": {
        "matrix_code": "swh_Latn", "iso3_code": "swa",
        "script": "Latn", "family": "Niger-Congo",
    },
}


@dataclass(frozen=True)
class PairForgetting:
    past_language: str
    incoming_language: str
    mean_forgetting: float
    observation_count: int
    order_ids: tuple[str, ...]


@dataclass(frozen=True)
class PriorMatrix:
    """A matrix normalized so larger scores always predict more forgetting."""

    name: str
    values: Mapping[str, Mapping[str, float]]
    value_kind: str
    language_codes: Mapping[str, str]

    def score(self, past_language: str, incoming_language: str) -> float:
        past_code = self._resolve_code(past_language)
        incoming_code = self._resolve_code(incoming_language)
        try:
            raw_score = float(self.values[past_code][incoming_code])
        except KeyError as exc:
            raise KeyError(
                f"Prior {self.name!r} has no score for "
                f"{past_language!r} → {incoming_language!r} "
                f"(matrix codes {past_code!r}, {incoming_code!r})."
            ) from exc
        if not math.isfinite(raw_score):
            raise ValueError(f"Prior {self.name!r} contains a non-finite score.")

        if self.value_kind == "paper_interference":
            return max(0.0, -raw_score)
        if self.value_kind == "distance":
            return max(0.0, raw_score)
        if self.value_kind == "similarity":
            return max(0.0, 1.0 - raw_score)
        if self.value_kind == "risk":
            return max(0.0, raw_score)
        raise ValueError(f"Unsupported prior value kind: {self.value_kind!r}.")

    def _resolve_code(self, language: str) -> str:
        metadata = LANGUAGE_METADATA.get(language, {})
        candidates = (
            self.language_codes.get(language),
            metadata.get("matrix_code"),
            metadata.get("iso3_code"),
            language,
        )
        for candidate in candidates:
            if candidate is not None and candidate in self.values:
                return candidate
        return next(candidate for candidate in candidates if candidate is not None)


def load_wide_matrix(path: str | Path) -> dict[str, dict[str, float]]:
    """Read a square CSV matrix whose first column contains row language codes."""
    matrix_path = Path(path)
    with matrix_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames
        if not headers or len(headers) < 3:
            raise ValueError(
                f"{matrix_path} must have a row-language column and at least "
                "two matrix columns."
            )
        row_header = headers[0]
        column_languages = headers[1:]
        if len(set(column_languages)) != len(column_languages):
            raise ValueError("Matrix column language codes must be unique.")

        matrix: dict[str, dict[str, float]] = {}
        for line_number, row in enumerate(reader, start=2):
            row_language = row.get(row_header)
            if not isinstance(row_language, str) or not row_language.strip():
                raise ValueError(f"Missing row language code on line {line_number}.")
            row_language = row_language.strip()
            if row_language in matrix:
                raise ValueError(f"Duplicate matrix row for {row_language!r}.")
            values = {}
            for column_language in column_languages:
                cell = row.get(column_language)
                try:
                    value = float(cell)  # type: ignore[arg-type]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value at {row_language!r}, "
                        f"{column_language!r}."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value at {row_language!r}, "
                        f"{column_language!r}."
                    )
                values[column_language] = value
            matrix[row_language] = values

    if set(matrix) != set(column_languages):
        raise ValueError(
            "Matrix must be square and use the same language codes for rows "
            "and columns."
        )
    return matrix


def make_builtin_prior_matrices(
    languages: Sequence[str],
) -> dict[str, PriorMatrix]:
    """Build the script and broad-family mismatch priors for known pilot languages."""
    unknown = sorted(set(languages) - set(LANGUAGE_METADATA))
    if unknown:
        raise ValueError(f"No script/family metadata is configured for {unknown}.")

    outputs = {}
    for name, metadata_key in (
        ("script_mismatch", "script"),
        ("family_mismatch", "family"),
    ):
        matrix = {
            left: {
                right: float(
                    LANGUAGE_METADATA[left][metadata_key]
                    != LANGUAGE_METADATA[right][metadata_key]
                )
                for right in languages
            }
            for left in languages
        }
        outputs[name] = PriorMatrix(
            name=name,
            values=matrix,
            value_kind="risk",
            language_codes={language: language for language in languages},
        )
    return outputs


def load_sequential_forgetting(
    result_paths: Sequence[str | Path],
    *,
    expected_method: str = "B0",
) -> tuple[list[dict[str, Any]], list[PairForgetting]]:
    """Extract each incoming-language effect from completed sequential records."""
    observations: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for result_path in sorted(Path(path) for path in result_paths):
        with result_path.open(encoding="utf-8") as source:
            record = json.load(source)
        if record.get("completed") is not True:
            raise ValueError(f"Incomplete result record: {result_path}.")
        if record.get("method_id") != expected_method:
            raise ValueError(
                f"Expected {expected_method} records; found "
                f"{record.get('method_id')!r} in {result_path}."
            )
        if record.get("evaluation_mode") != "sequential":
            raise ValueError(f"Expected a sequential result record: {result_path}.")
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or run_id in seen_run_ids:
            raise ValueError(f"Missing or duplicate run_id in {result_path}.")
        seen_run_ids.add(run_id)

        task_order = record.get("task_order")
        performance = record.get("performance_matrix")
        if (
            not isinstance(task_order, list)
            or not task_order
            or len(set(task_order)) != len(task_order)
            or not isinstance(performance, list)
            or len(performance) != len(task_order)
            or any(
                not isinstance(row, list) or len(row) != len(task_order)
                for row in performance
            )
        ):
            raise ValueError(f"Invalid performance matrix in {result_path}.")
        order_id = record.get("order_id")
        if not isinstance(order_id, str) or not order_id:
            raise ValueError(f"Missing order_id in {result_path}.")

        normalized_matrix: list[list[float]] = []
        for row in performance:
            normalized_row = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Non-numeric accuracy in {result_path}.")
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite accuracy in {result_path}.")
                normalized_row.append(float(value))
            normalized_matrix.append(normalized_row)

        for stage_index in range(1, len(task_order)):
            incoming_language = task_order[stage_index]
            for past_index in range(stage_index):
                past_language = task_order[past_index]
                before = normalized_matrix[stage_index - 1][past_index]
                after = normalized_matrix[stage_index][past_index]
                observations.append({
                    "run_id": run_id,
                    "order_id": order_id,
                    "stage_index": stage_index,
                    "past_language": past_language,
                    "incoming_language": incoming_language,
                    "accuracy_before": before,
                    "accuracy_after": after,
                    "forgetting": before - after,
                })

    if not observations:
        raise ValueError("No previous-language forgetting observations were found.")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[
            (observation["past_language"], observation["incoming_language"])
        ].append(observation)
    pair_forgetting = []
    for (past_language, incoming_language), pair_observations in sorted(grouped.items()):
        pair_forgetting.append(PairForgetting(
            past_language=past_language,
            incoming_language=incoming_language,
            mean_forgetting=(
                sum(item["forgetting"] for item in pair_observations)
                / len(pair_observations)
            ),
            observation_count=len(pair_observations),
            order_ids=tuple(sorted({item["order_id"] for item in pair_observations})),
        ))
    return observations, pair_forgetting


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Calculate Spearman's rho with average ranks for tied values."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    covariance = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    left_variance = sum((rank - left_mean) ** 2 for rank in left_ranks)
    right_variance = sum((rank - right_mean) ** 2 for rank in right_ranks)
    if left_variance == 0 or right_variance == 0:
        return None
    return covariance / math.sqrt(left_variance * right_variance)


def make_composite_prior(
    name: str,
    components: Sequence[PriorMatrix],
    languages: Sequence[str],
) -> PriorMatrix:
    """Average component percentile ranks without fitting weights to outcomes."""
    if len(components) < 2:
        raise ValueError("A composite prior needs at least two components.")
    if len(set(languages)) < 2:
        raise ValueError("A composite prior needs at least two languages.")

    directed_pairs = [
        (past, incoming)
        for past in languages
        for incoming in languages
        if past != incoming
    ]
    normalized_components = []
    for component in components:
        raw_scores = [
            component.score(past, incoming)
            for past, incoming in directed_pairs
        ]
        ranks = _average_ranks(raw_scores)
        maximum_rank = max(ranks)
        minimum_rank = min(ranks)
        if maximum_rank == minimum_rank:
            normalized = [0.5] * len(ranks)
        else:
            normalized = [
                (rank - minimum_rank) / (maximum_rank - minimum_rank)
                for rank in ranks
            ]
        normalized_components.append(normalized)

    values = {language: {language: 0.0} for language in languages}
    for pair_index, (past, incoming) in enumerate(directed_pairs):
        values[past][incoming] = sum(
            component[pair_index] for component in normalized_components
        ) / len(normalized_components)
    return PriorMatrix(
        name=name,
        values=values,
        value_kind="risk",
        language_codes={language: language for language in languages},
    )


def _permuted_spearman(
    pairs: Sequence[PairForgetting],
    prior: PriorMatrix,
    label_mapping: Mapping[str, str],
) -> float | None:
    scores = [
        prior.score(label_mapping[pair.past_language],
                    label_mapping[pair.incoming_language])
        for pair in pairs
    ]
    observed = [pair.mean_forgetting for pair in pairs]
    return spearman_correlation(scores, observed)


def label_permutation_test(
    pairs: Sequence[PairForgetting],
    prior: PriorMatrix,
    *,
    exact_limit: int = 8,
    monte_carlo_repetitions: int = 10000,
    seed: int = 20261116,
) -> dict[str, Any]:
    """Test positive association by permuting language labels on the prior matrix."""
    languages = sorted({
        language
        for pair in pairs
        for language in (pair.past_language, pair.incoming_language)
    })
    if len(languages) < 2:
        return {
            "p_one_sided": None,
            "permutation_count": 0,
            "test_method": "not_testable",
        }

    observed = _permuted_spearman(
        pairs, prior, {language: language for language in languages}
    )
    if observed is None:
        return {
            "p_one_sided": None,
            "permutation_count": 0,
            "test_method": "not_testable",
        }

    exact = len(languages) <= exact_limit
    if exact:
        label_permutations = permutations(languages)
    else:
        rng = random.Random(seed)
        label_permutations = (
            tuple(rng.sample(languages, len(languages)))
            for _ in range(monte_carlo_repetitions)
        )

    extreme_count = 0
    valid_count = 0
    for permuted_labels in label_permutations:
        mapping = dict(zip(languages, permuted_labels))
        rho = _permuted_spearman(pairs, prior, mapping)
        if rho is None:
            continue
        valid_count += 1
        if rho >= observed - 1e-12:
            extreme_count += 1
    if valid_count == 0:
        p_value = None
    elif exact:
        p_value = extreme_count / valid_count
    else:
        p_value = (extreme_count + 1) / (valid_count + 1)
    return {
        "p_one_sided": p_value,
        "permutation_count": valid_count,
        "test_method": (
            "exact_language_label_permutation"
            if exact
            else "monte_carlo_language_label_permutation"
        ),
    }


def _holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Apply Holm's step-down correction while preserving input order."""
    valid = [
        (index, value)
        for index, value in enumerate(p_values)
        if value is not None
    ]
    adjusted: list[float | None] = [None] * len(p_values)
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[index] = running_max
    return adjusted


def compare_priors(
    pair_forgetting: Sequence[PairForgetting],
    priors: Mapping[str, PriorMatrix],
) -> list[dict[str, Any]]:
    """Correlate priors with forgetting and test association by label permutation."""
    output = []
    for prior_name, prior in priors.items():
        pairs = []
        scores = []
        observed = []
        for pair in pair_forgetting:
            try:
                score = prior.score(pair.past_language, pair.incoming_language)
            except KeyError:
                continue
            pairs.append(pair)
            scores.append(score)
            observed.append(pair.mean_forgetting)
        output.append({
            "prior": prior_name,
            "spearman_rho": spearman_correlation(scores, observed),
            "directed_language_pairs": len(pairs),
            "forgetting_observations": sum(
                pair.observation_count for pair in pairs
            ),
            "order_count": len({
                order_id for pair in pairs for order_id in pair.order_ids
            }),
            **label_permutation_test(pairs, prior),
        })
    adjusted = _holm_adjust([
        row["p_one_sided"] for row in output
    ])
    for row, adjusted_p in zip(output, adjusted):
        row["p_holm"] = adjusted_p
    return output


def compare_priors_by_order(
    observations: Sequence[Mapping[str, Any]],
    priors: Mapping[str, PriorMatrix],
) -> list[dict[str, Any]]:
    """Report descriptive rank correlations separately for each stream order."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation["order_id"])].append(observation)
    rows = []
    for order_id, order_observations in sorted(grouped.items()):
        for prior_name, prior in priors.items():
            scores = []
            observed = []
            for item in order_observations:
                try:
                    scores.append(prior.score(
                        str(item["past_language"]),
                        str(item["incoming_language"]),
                    ))
                except KeyError:
                    continue
                observed.append(float(item["forgetting"]))
            rows.append({
                "order_id": order_id,
                "prior": prior_name,
                "spearman_rho": spearman_correlation(scores, observed),
                "directed_language_pairs": len(scores),
            })
    return rows


def random_prior_control(
    pair_forgetting: Sequence[PairForgetting],
    *,
    repetitions: int = 1000,
    seed: int = 20261116,
) -> dict[str, Any]:
    """Estimate a random-prior correlation reference distribution."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if len(pair_forgetting) < 2:
        return {
            "repetitions": repetitions,
            "valid_repetitions": 0,
            "mean_rho": None,
            "std_rho": None,
            "rho_2_5_percentile": None,
            "rho_97_5_percentile": None,
        }
    rng = random.Random(seed)
    observed = [pair.mean_forgetting for pair in pair_forgetting]
    correlations = []
    for _ in range(repetitions):
        random_scores = [rng.random() for _ in pair_forgetting]
        rho = spearman_correlation(random_scores, observed)
        if rho is not None:
            correlations.append(rho)
    if not correlations:
        return {
            "repetitions": repetitions,
            "valid_repetitions": 0,
            "mean_rho": None,
            "std_rho": None,
            "rho_2_5_percentile": None,
            "rho_97_5_percentile": None,
        }
    sorted_correlations = sorted(correlations)

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(sorted_correlations) - 1))
        return sorted_correlations[index]

    mean_rho = sum(correlations) / len(correlations)
    variance = sum((value - mean_rho) ** 2 for value in correlations) / len(correlations)
    return {
        "repetitions": repetitions,
        "valid_repetitions": len(correlations),
        "mean_rho": mean_rho,
        "std_rho": math.sqrt(variance),
        "rho_2_5_percentile": percentile(0.025),
        "rho_97_5_percentile": percentile(0.975),
    }


def write_analysis_outputs(
    output_dir: str | Path,
    observations: Sequence[Mapping[str, Any]],
    pair_forgetting: Sequence[PairForgetting],
    correlations: Sequence[Mapping[str, Any]],
    order_correlations: Sequence[Mapping[str, Any]],
    random_control: Mapping[str, Any],
    *,
    missing_priors: Sequence[str] = (),
    excluded_priors: Mapping[str, str] | None = None,
) -> None:
    """Write transparent CSV/JSON outputs for a Phase 3 analysis run."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    observation_fields = (
        "run_id", "order_id", "stage_index", "past_language",
        "incoming_language", "accuracy_before", "accuracy_after", "forgetting",
    )
    with (destination / "forgetting_observations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=observation_fields)
        writer.writeheader()
        writer.writerows(observations)

    with (destination / "pair_forgetting.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "past_language", "incoming_language", "mean_forgetting",
                "observation_count", "order_ids",
            ),
        )
        writer.writeheader()
        for pair in pair_forgetting:
            writer.writerow({
                "past_language": pair.past_language,
                "incoming_language": pair.incoming_language,
                "mean_forgetting": pair.mean_forgetting,
                "observation_count": pair.observation_count,
                "order_ids": "|".join(pair.order_ids),
            })

    with (destination / "prior_correlations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fields = (
            "prior", "spearman_rho", "p_one_sided", "p_holm",
            "permutation_count", "test_method", "directed_language_pairs",
            "forgetting_observations", "order_count",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correlations)

    with (destination / "order_correlations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fields = (
            "order_id", "prior", "spearman_rho", "directed_language_pairs",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(order_correlations)

    (destination / "summary.json").write_text(
        json.dumps({
            "analysis": "phase3_rq1_prior_vs_forgetting",
            "forgetting_definition": (
                "accuracy before an incoming-language stage minus accuracy "
                "after that stage, for each previously seen language"
            ),
            "pair_aggregation": (
                "mean forgetting per directed past-language/incoming-language "
                "pair across stream orders"
            ),
            "permutation_test": (
                "one-sided positive Spearman test by permuting language labels "
                "on each complete prior matrix; Holm correction covers tested "
                "priors. This is conditional on the observed pilot forgetting "
                "matrix and does not capture training-seed uncertainty."
            ),
            "correlations": list(correlations),
            "per_order_correlations": list(order_correlations),
            "random_prior_control": dict(random_control),
            "missing_prior_inputs": list(missing_priors),
            "excluded_priors": dict(excluded_priors or {}),
            "interpretation_note": (
                "Permutation p-values test alignment of prior language labels "
                "with this pilot's observed pair-forgetting matrix; they are "
                "not evidence of robustness across training seeds. Interpret "
                "them with the per-order results and replicate with additional "
                "training seeds before making a general claim."
            ),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
