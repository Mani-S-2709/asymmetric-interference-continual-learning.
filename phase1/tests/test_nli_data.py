import pytest

from continual_learning.nli_data import (
    format_nli_example,
    prepare_nli_stream,
    select_source_indices,
    selection_sha256,
)
from continual_learning.stream_loader import (
    HuggingFaceStreamLoader,
    StreamStage,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []
        self._fingerprint = "test-fingerprint"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])

    def map(self, function, *, remove_columns, desc):
        return FakeDataset([function(row) for row in self.rows])


@pytest.mark.parametrize(
    ("label", "target"),
    [
        (0, " entailment"),
        (1, " neutral"),
        (2, " contradiction"),
        ("entailment", " entailment"),
        ("neutral", " neutral"),
        ("contradiction", " contradiction"),
    ],
)
def test_format_nli_example_normalizes_label_forms(label, target):
    formatted = format_nli_example(
        {
            "premise": "premise text",
            "hypothesis": "hypothesis text",
            "label": label,
        }
    )

    assert formatted["target"] == target
    assert "Premise: premise text" in formatted["prompt"]


def test_format_nli_example_rejects_nonstandard_label():
    with pytest.raises(ValueError, match="label ID"):
        format_nli_example(
            {"premise": "p", "hypothesis": "h", "label": 3}
        )


def test_sample_selection_is_reproducible_and_records_exact_order():
    first = select_source_indices(100, 12, 42)
    second = select_source_indices(100, 12, 42)

    assert first == second
    assert len(set(first)) == 12
    assert selection_sha256(first) == selection_sha256(second)
    assert first != select_source_indices(100, 12, 43)


def test_prepare_nli_stream_supports_mixed_sources_and_provenance():
    stages = [
        StreamStage("en", "nli", dataset_id="facebook/xnli", dataset_config="en"),
        StreamStage(
            "hi",
            "nli",
            dataset_id="AdaMLLab/indicxnli_repaired",
            dataset_config="hi",
            revision="pinned",
        ),
        StreamStage(
            "bn",
            "nli",
            dataset_id="AdaMLLab/indicxnli_repaired",
            dataset_config="bn",
            revision="pinned",
        ),
    ]
    calls = []

    def load_dataset(dataset_id, *args, split, **kwargs):
        config = args[0] if args else None
        calls.append((dataset_id, config, split, kwargs.get("revision")))
        rows = [
            {
                "premise": f"{dataset_id} {config} {split} premise {index}",
                "hypothesis": f"hypothesis {index}",
                "label": index % 3,
            }
            for index in range(10 if split == "train" else 5)
        ]
        return FakeDataset(rows)

    loader = HuggingFaceStreamLoader(
        stages,
        shuffle_train=False,
        dataset_loader=load_dataset,
    )
    train_by_seed, test_by_language, manifest = prepare_nli_stream(
        stages,
        loader,
        experiment_id="test-mixed-nli",
        seeds=[42, 43],
        train_count=4,
        test_count=2,
        test_seed=99,
        require_parallel_label_alignment=True,
    )

    assert list(train_by_seed[42]) == ["en", "hi", "bn"]
    assert list(test_by_language) == ["en", "hi", "bn"]
    assert set(train_by_seed[42]["bn"][0]) == {"prompt", "target"}
    assert test_by_language["bn"][0]["label"] in (0, 1, 2)
    assert manifest["sources"]["bn"]["dataset_id"] == (
        "AdaMLLab/indicxnli_repaired"
    )
    assert manifest["sources"]["bn"]["train_selection_by_seed"]["42"]["indices"] == (
        manifest["sources"]["en"]["train_selection_by_seed"]["42"]["indices"]
    )
    assert manifest["sources"]["bn"]["test_selection"]["indices"] == (
        manifest["sources"]["en"]["test_selection"]["indices"]
    )
    assert manifest["require_parallel_label_alignment"] is True
    assert manifest["data_manifest_sha256"]
    assert len(calls) == 6


def test_prepare_nli_stream_rejects_misaligned_language_labels():
    stages = [
        StreamStage("en", "nli", dataset_id="source/en"),
        StreamStage("bn", "nli", dataset_id="source/bn"),
    ]

    def load_misaligned(dataset_id, *, split, **kwargs):
        language_offset = 1 if dataset_id == "source/bn" else 0
        rows = [
            {
                "premise": f"{dataset_id} premise {index}",
                "hypothesis": f"hypothesis {index}",
                "label": (index + language_offset) % 3,
            }
            for index in range(6)
        ]
        return FakeDataset(rows)

    loader = HuggingFaceStreamLoader(stages, dataset_loader=load_misaligned)

    with pytest.raises(ValueError, match="not parallel-label aligned"):
        prepare_nli_stream(
            stages,
            loader,
            experiment_id="misaligned-test",
            seeds=[42],
            train_count=3,
            test_count=2,
            test_seed=7,
            require_parallel_label_alignment=True,
        )
