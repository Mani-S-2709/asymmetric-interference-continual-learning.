import sys
from types import SimpleNamespace

import pytest

from phase4.interference_replay import AsymmetricInterferenceReplaySelector
from phase4.run_phase4 import (
    DEFAULT_PROTOCOL,
    _prepare_m3_stage_datasets,
    run_selected_pair,
)
from shared.continual_learning_lora import ReplayBuffer


class _ListDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def shuffle(self, seed):
        return self


def test_m3_stage_data_uses_fixed_replay_ratio_and_only_prior_languages(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=_ListDataset))
    train_sets = {
        language: [{"id": f"{language}-{index}"} for index in range(2)]
        for language in ("en", "de", "fr")
    }
    selector = AsymmetricInterferenceReplaySelector(
        {
            "en": {"de": -0.5, "fr": -0.2},
            "de": {"de": 0.0, "fr": -0.8},
            "fr": {"de": -0.1, "fr": 0.0},
        },
        seed=3,
    )

    datasets, stages = _prepare_m3_stage_datasets(
        ("en", "de", "fr"),
        train_sets,
        selector,
        capacity=2,
        replay_seed=11,
        training_seed=42,
        replay_ratio=1,
    )

    assert [stages[index]["replay_examples"] for index in range(3)] == [0, 2, 2]
    assert len(datasets["en"]) == 2
    assert len(datasets["de"]) == 4
    assert len(datasets["fr"]) == 4
    assert set(stages[1]["replay_examples_by_language"]) <= {"en"}
    assert set(stages[2]["replay_examples_by_language"]) <= {"en", "de"}


def test_replay_buffer_snapshot_does_not_expose_internal_example_dict():
    buffer = ReplayBuffer(2, seed=1)
    buffer.add("en", {"id": 1})

    snapshot = buffer.snapshot()
    snapshot[0].example["id"] = 99

    assert buffer.snapshot()[0].example["id"] == 1


def test_phase4_runner_blocks_before_data_preparation_until_prior_selection():
    with pytest.raises(RuntimeError, match="matching-roster Phase 3 v2 analysis"):
        run_selected_pair(
            method_id="B0",
            order_id="order-1",
            protocol_path=DEFAULT_PROTOCOL,
        )
