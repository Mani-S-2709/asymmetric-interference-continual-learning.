import math

import pytest

from phase4.interference_replay import AsymmetricInterferenceReplaySelector
from shared.continual_learning_lora.replay import ReplayItem


def _item(language: str, index: int) -> ReplayItem:
    return ReplayItem(language=language, example={"id": index})


def test_language_weights_follow_past_row_and_incoming_column():
    selector = AsymmetricInterferenceReplaySelector(
        {
            "en": {"de": -0.8, "hi": -0.1},
            "hi": {"de": -0.2, "hi": -0.9},
        }
    )

    assert selector.language_weights(["en", "hi"], "de") == pytest.approx(
        {"en": 0.8, "hi": 0.2}
    )
    assert selector.language_weights(["en", "hi"], "hi") == pytest.approx(
        {"en": 0.1, "hi": 0.9}
    )


def test_nonnegative_scores_clip_to_zero_and_use_uniform_fallback():
    selector = AsymmetricInterferenceReplaySelector(
        {"en": {"de": 0.4}, "hi": {"de": 0.0}}
    )

    assert selector.language_weights(["en", "hi"], "de") == {
        "en": 0.5,
        "hi": 0.5,
    }


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({}, "No interference scores"),
        ({"en": {}}, "Missing interference score"),
        ({"en": {"de": math.nan}}, "must be finite"),
        ({"en": {"de": math.inf}}, "must be finite"),
        ({"en": {"de": 10**400}}, "must be finite"),
        ({"en": {"de": "not-a-number"}}, "must be a finite number"),
    ],
)
def test_missing_or_invalid_matrix_values_raise(scores, message):
    selector = AsymmetricInterferenceReplaySelector(scores)

    with pytest.raises(ValueError, match=message):
        selector.language_weights(["en"], "de")


def test_select_returns_distinct_candidates_and_prefers_harmful_language():
    examples = [_item("en", 0), _item("hi", 1), _item("hi", 2)]
    selector = AsymmetricInterferenceReplaySelector(
        {"en": {"de": 0.0}, "hi": {"de": -1.0}},
        seed=42,
    )

    selected = selector.select(examples, 1, incoming_language="de")

    assert selected == [examples[1]] or selected == [examples[2]]
    assert len({id(item) for item in selected}) == len(selected)


def test_select_rejects_negative_budget():
    selector = AsymmetricInterferenceReplaySelector({})

    with pytest.raises(ValueError, match="budget cannot be negative"):
        selector.select([], -1, incoming_language="de")
