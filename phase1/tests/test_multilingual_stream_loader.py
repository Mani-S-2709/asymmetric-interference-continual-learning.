from continual_learning.stream_loader import (
    HuggingFaceStreamLoader,
    StreamStage,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


def test_loader_routes_each_language_to_its_declared_source():
    calls = []

    def load_dataset(dataset_id, *args, **kwargs):
        calls.append((dataset_id, args, kwargs))
        return FakeDataset([{"label": 0}, {"label": 1}])

    stages = [
        StreamStage(
            language="en",
            task="nli",
            dataset_id="facebook/xnli",
            dataset_config="en",
            revision="xnli-revision",
        ),
        StreamStage(
            language="hi",
            task="nli",
            dataset_id="AdaMLLab/indicxnli_repaired",
            dataset_config="hi",
            revision="indicxnli-revision",
        ),
        StreamStage(
            language="bn",
            task="nli",
            dataset_id="AdaMLLab/indicxnli_repaired",
            dataset_config="bn",
            revision="indicxnli-revision",
        ),
    ]
    loader = HuggingFaceStreamLoader(
        stages,
        max_samples_per_stage=1,
        dataset_loader=load_dataset,
    )

    loaded = [loader.load_stage(stage) for stage in stages]

    assert [len(dataset) for dataset in loaded] == [1, 1, 1]
    assert calls == [
        ("facebook/xnli", ("en",), {
            "split": "train",
            "revision": "xnli-revision",
        }),
        ("AdaMLLab/indicxnli_repaired", ("hi",), {
            "split": "train",
            "revision": "indicxnli-revision",
        }),
        ("AdaMLLab/indicxnli_repaired", ("bn",), {
            "split": "train",
            "revision": "indicxnli-revision",
        }),
    ]


def test_loader_supports_dataset_without_a_configuration():
    calls = []

    def load_dataset(dataset_id, *args, **kwargs):
        calls.append((dataset_id, args, kwargs))
        return FakeDataset([{"label": 2}])

    stage = StreamStage(
        language="xyz",
        task="nli",
        dataset_id="organization/custom-nli",
    )
    loader = HuggingFaceStreamLoader(
        [stage],
        dataset_loader=load_dataset,
    )

    loaded = loader.load_stage(stage)

    assert loaded.rows == [{"label": 2}]
    assert calls == [
        ("organization/custom-nli", (), {"split": "train"}),
    ]
