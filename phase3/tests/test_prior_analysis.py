import json

import pytest

from phase3.prior_analysis import (
    PairForgetting,
    PriorMatrix,
    _holm_adjust,
    compare_priors,
    compare_priors_by_order,
    label_permutation_test,
    load_sequential_forgetting,
    load_wide_matrix,
    make_composite_prior,
    make_builtin_prior_matrices,
    random_prior_control,
    spearman_correlation,
    write_analysis_outputs,
)


def test_published_interference_uses_paper_row_column_and_sign():
    matrix = {
        "eng_Latn": {"hin_Deva": -0.8, "spa_Latn": -0.2},
        "hin_Deva": {"eng_Latn": -0.1, "spa_Latn": -0.3},
    }
    prior = PriorMatrix(
        name="published",
        values=matrix,
        value_kind="paper_interference",
        language_codes={"en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn"},
    )

    assert prior.score("en", "hi") == pytest.approx(0.8)
    assert prior.score("hi", "en") == pytest.approx(0.1)


def test_prior_accepts_matrix_rows_in_project_iso_codes():
    prior = PriorMatrix(
        name="published",
        values={"en": {"hi": -0.5}, "hi": {"en": -0.2}},
        value_kind="paper_interference",
        language_codes={"en": "eng_Latn", "hi": "hin_Deva"},
    )

    assert prior.score("en", "hi") == pytest.approx(0.5)


def test_prior_accepts_uriel_iso3_codes():
    prior = PriorMatrix(
        name="uriel",
        values={"eng": {"hin": 0.5508}, "hin": {"eng": 0.5508}},
        value_kind="distance",
        language_codes={},
    )

    assert prior.score("en", "hi") == pytest.approx(0.5508)


def test_matrix_loader_requires_square_finite_unique_matrix(tmp_path):
    matrix_path = tmp_path / "matrix.csv"
    matrix_path.write_text(
        "language,en,hi\n"
        "en,0,0.2\n"
        "hi,0.2,0\n",
        encoding="utf-8",
    )

    assert load_wide_matrix(matrix_path) == {
        "en": {"en": 0.0, "hi": 0.2},
        "hi": {"en": 0.2, "hi": 0.0},
    }

    matrix_path.write_text(
        "language,en,hi\n"
        "en,0,nan\n"
        "hi,0.2,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Non-finite"):
        load_wide_matrix(matrix_path)


def test_forgetting_is_before_minus_after_for_each_past_language(tmp_path):
    record = {
        "completed": True,
        "method_id": "B0",
        "evaluation_mode": "sequential",
        "run_id": "b0-order-1",
        "order_id": "order-1",
        "task_order": ["en", "hi", "de"],
        "performance_matrix": [
            [0.50, 0.40, 0.35],
            [0.45, 0.52, 0.38],
            [0.40, 0.50, 0.55],
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(record), encoding="utf-8")

    observations, pairs = load_sequential_forgetting([result_path])
    by_pair = {
        (pair.past_language, pair.incoming_language): pair.mean_forgetting
        for pair in pairs
    }

    assert len(observations) == 3
    assert by_pair[("en", "hi")] == pytest.approx(0.05)
    assert by_pair[("en", "de")] == pytest.approx(0.05)
    assert by_pair[("hi", "de")] == pytest.approx(0.02)


def test_pair_aggregation_and_prior_comparison_use_directed_pairs(tmp_path):
    records = [
        {
            "completed": True,
            "method_id": "B0",
            "evaluation_mode": "sequential",
            "run_id": f"b0-{order}",
            "order_id": order,
            "task_order": ["en", "hi"],
            "performance_matrix": [[0.7, 0.4], [after, 0.5]],
        }
        for order, after in (("order-1", 0.5), ("order-2", 0.6))
    ]
    paths = []
    for index, record in enumerate(records):
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    _, pairs = load_sequential_forgetting(paths)
    prior = PriorMatrix(
        name="risk",
        values={"en": {"hi": 1.0}, "hi": {"en": 0.0}},
        value_kind="risk",
        language_codes={"en": "en", "hi": "hi"},
    )

    assert len(pairs) == 1
    assert pairs[0].mean_forgetting == pytest.approx(0.15)
    assert pairs[0].observation_count == 2
    assert compare_priors(pairs, {"risk": prior})[0]["directed_language_pairs"] == 1


def test_spearman_handles_ties_and_constant_inputs():
    assert spearman_correlation([1, 1, 3], [3, 3, 9]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1], [1, 2]) is None


def test_composite_prior_uses_equal_weight_percentile_ranks():
    builtins = make_builtin_prior_matrices(["en", "hi", "es", "de", "sw"])
    uriel = PriorMatrix(
        name="uriel",
        values={
            "en": {"en": 0.0, "hi": 0.7, "es": 0.2, "de": 0.3, "sw": 0.9},
            "hi": {"en": 0.7, "hi": 0.0, "es": 0.8, "de": 0.6, "sw": 0.5},
            "es": {"en": 0.2, "hi": 0.8, "es": 0.0, "de": 0.1, "sw": 0.9},
            "de": {"en": 0.3, "hi": 0.6, "es": 0.1, "de": 0.0, "sw": 0.8},
            "sw": {"en": 0.9, "hi": 0.5, "es": 0.9, "de": 0.8, "sw": 0.0},
        },
        value_kind="distance",
        language_codes={language: language for language in ["en", "hi", "es", "de", "sw"]},
    )

    composite = make_composite_prior(
        "script+uriel",
        [builtins["script_mismatch"], uriel],
        ["en", "hi", "es", "de", "sw"],
    )

    assert 0.0 <= composite.score("en", "hi") <= 1.0
    assert composite.score("en", "hi") == composite.score("hi", "en")


def test_label_permutation_test_is_exact_for_pilot_language_count():
    pairs = [
        PairForgetting("en", "hi", 0.3, 1, ("order-1",)),
        PairForgetting("en", "es", 0.2, 1, ("order-2",)),
        PairForgetting("hi", "de", 0.1, 1, ("order-3",)),
        PairForgetting("es", "sw", 0.4, 1, ("order-4",)),
    ]
    languages = ("en", "hi", "es", "de", "sw")
    prior = PriorMatrix(
        name="risk",
        values={
            left: {
                right: float((left_index * 3 + right_index) % 5)
                for right_index, right in enumerate(languages)
            }
            for left_index, left in enumerate(languages)
        },
        value_kind="risk",
        language_codes={language: language for language in languages},
    )

    result = label_permutation_test(pairs, prior)

    assert result["test_method"] == "exact_language_label_permutation"
    assert result["permutation_count"] == 120
    assert 0.0 < result["p_one_sided"] <= 1.0


def test_holm_adjustment_preserves_order_and_handles_missing_tests():
    adjusted = _holm_adjust([0.01, 0.04, 0.03, None])

    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert adjusted[3] is None


def test_per_order_correlations_are_reported_separately():
    observations = [
        {
            "order_id": "order-1",
            "past_language": "en",
            "incoming_language": "hi",
            "forgetting": 0.2,
        },
        {
            "order_id": "order-1",
            "past_language": "sw",
            "incoming_language": "hi",
            "forgetting": 0.1,
        },
    ]
    prior = make_builtin_prior_matrices(["en", "hi", "sw"])["script_mismatch"]

    rows = compare_priors_by_order(observations, {"script": prior})

    assert len(rows) == 1
    assert rows[0]["order_id"] == "order-1"
    assert rows[0]["forgetting_observations"] == 2


def test_per_order_count_is_observations_across_seeds():
    observations = [
        {
            "order_id": "order-1",
            "past_language": "en",
            "incoming_language": "hi",
            "forgetting": value,
        }
        for value in (0.1, 0.2)
    ] + [
        {
            "order_id": "order-1",
            "past_language": "hi",
            "incoming_language": "en",
            "forgetting": value,
        }
        for value in (0.2, 0.3)
    ]
    prior = PriorMatrix(
        name="risk",
        values={"en": {"en": 0.0, "hi": 1.0}, "hi": {"en": 0.0, "hi": 0.0}},
        value_kind="risk",
        language_codes={"en": "en", "hi": "hi"},
    )

    rows = compare_priors_by_order(observations, {"risk": prior})

    assert rows[0]["forgetting_observations"] == 4
    assert "directed_language_pairs" not in rows[0]


def test_script_and_family_priors_match_pilot_metadata():
    priors = make_builtin_prior_matrices(["en", "hi", "es", "de", "sw"])

    assert priors["script_mismatch"].score("en", "hi") == 1.0
    assert priors["script_mismatch"].score("en", "sw") == 0.0
    assert priors["family_mismatch"].score("en", "hi") == 0.0
    assert priors["family_mismatch"].score("en", "sw") == 1.0


def test_v2_script_and_family_priors_cover_new_languages():
    languages = ["en", "hi", "es", "de", "sw", "ar", "ru", "tr", "th", "zh"]
    priors = make_builtin_prior_matrices(languages)

    assert priors["script_mismatch"].score("en", "ru") == 1.0
    assert priors["script_mismatch"].score("en", "tr") == 0.0
    assert priors["family_mismatch"].score("en", "ru") == 0.0
    assert priors["family_mismatch"].score("en", "ar") == 1.0


def test_ten_language_label_test_uses_monte_carlo_permutations():
    languages = ("en", "hi", "es", "de", "sw", "ar", "ru", "tr", "th", "zh")
    pairs = [
        PairForgetting(
            past,
            incoming,
            float((left_index * 3 + right_index) % 17),
            1,
            ("order-1",),
        )
        for left_index, past in enumerate(languages)
        for right_index, incoming in enumerate(languages)
        if past != incoming
    ]
    prior = PriorMatrix(
        name="risk",
        values={
            left: {
                right: float((left_index * 7 + right_index * 2) % 19)
                for right_index, right in enumerate(languages)
            }
            for left_index, left in enumerate(languages)
        },
        value_kind="risk",
        language_codes={language: language for language in languages},
    )

    result = label_permutation_test(
        pairs,
        prior,
        monte_carlo_repetitions=30,
        seed=7,
    )

    assert result["test_method"] == "monte_carlo_language_label_permutation"
    assert result["permutation_count"] == 30
    assert 0.0 < result["p_one_sided"] <= 1.0


def test_random_control_is_seed_reproducible():
    pairs = [
        PairForgetting("en", "hi", 0.1, 1, ("order-1",)),
        PairForgetting("hi", "de", 0.2, 1, ("order-1",)),
        PairForgetting("en", "sw", 0.3, 1, ("order-2",)),
    ]

    first = random_prior_control(pairs, repetitions=30, seed=7)
    second = random_prior_control(pairs, repetitions=30, seed=7)

    assert first == second
    assert first["valid_repetitions"] == 30


def test_analysis_summary_distinguishes_excluded_priors_from_missing_inputs(
    tmp_path,
):
    write_analysis_outputs(
        tmp_path,
        [],
        [],
        [],
        [],
        {},
        excluded_priors={
            "grambank_distance": "Unavailable for this pilot.",
        },
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["missing_prior_inputs"] == []
    assert summary["excluded_priors"] == {
        "grambank_distance": "Unavailable for this pilot.",
    }
