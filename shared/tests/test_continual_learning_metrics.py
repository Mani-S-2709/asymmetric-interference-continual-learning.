import csv
import json

import pytest

from shared.continual_learning_metrics import (
    EvaluationLedger,
    WandbRunLogger,
    average_accuracy,
    backward_transfer,
    calculate_metrics,
    final_performance,
    forgetting,
    forward_transfer,
)


MATRIX = [
    [0.80, 0.40, 0.30],
    [0.70, 0.75, 0.45],
    [0.65, 0.60, 0.80],
]
BASELINES = [0.50, 0.35, 0.25]


def test_standard_metrics_match_expected_values():
    assert average_accuracy(MATRIX) == pytest.approx(0.6833333333)
    assert final_performance(MATRIX) == pytest.approx(0.6833333333)
    assert backward_transfer(MATRIX) == pytest.approx(-0.15)
    assert forward_transfer(MATRIX, BASELINES) == pytest.approx(0.125)
    assert forgetting(MATRIX) == pytest.approx(0.15)


def test_calculate_metrics_exposes_common_names_and_aliases():
    result = calculate_metrics(MATRIX, BASELINES)

    assert result == {
        "ACC": pytest.approx(0.6833333333),
        "AA": pytest.approx(0.6833333333),
        "BWT": pytest.approx(-0.15),
        "FWT": pytest.approx(0.125),
        "F": pytest.approx(0.15),
        "FP": pytest.approx(0.6833333333),
    }


def test_single_task_returns_defined_average_only():
    assert calculate_metrics([[0.7]]) == {
        "ACC": 0.7,
        "AA": 0.7,
        "BWT": None,
        "FWT": None,
        "F": None,
        "FP": 0.7,
    }


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [[0.5, 0.6]],
        [[0.5, float("nan")], [0.4, 0.6]],
        [[0.5, 1.1], [0.4, 0.6]],
        [[True]],
    ],
)
def test_metrics_reject_invalid_matrices(matrix):
    with pytest.raises((TypeError, ValueError)):
        average_accuracy(matrix)


def test_forward_transfer_requires_one_baseline_per_task():
    with pytest.raises(ValueError, match="one zero-shot accuracy"):
        forward_transfer(MATRIX, [0.4])


def _make_ledger(tmp_path, tracker=None):
    return EvaluationLedger(
        task_order=["en", "hi", "de"],
        experiment_id="xnli-seed-42",
        output_dir=tmp_path,
        run_metadata={
            "phase": 1,
            "method": "sequential_finetuning",
            "seed": 42,
            "model": "Llama-3.2-1B",
        },
        tracker=tracker,
    )


def _record_all(ledger):
    for task_id, baseline in zip(ledger.task_order, BASELINES):
        ledger.record_baseline(
            task_id,
            baseline,
            language=task_id,
            task="xnli",
            split="test",
            dataset="xnli",
            num_examples=100,
        )

    for stage_index, row in enumerate(MATRIX):
        for task_id, accuracy in zip(ledger.task_order, row):
            ledger.record_evaluation(
                stage_index,
                task_id,
                accuracy,
                language=task_id,
                task="xnli",
                split="test",
                dataset="xnli",
                num_examples=100,
                metadata={"accuracy_unit": "fraction"},
            )


def test_ledger_logs_baselines_evaluations_and_metrics(tmp_path):
    ledger = _make_ledger(tmp_path)
    _record_all(ledger)

    assert ledger.performance_matrix() == MATRIX
    assert ledger.baseline_accuracies() == BASELINES
    assert ledger.compute_metrics()["F"] == pytest.approx(0.15)

    payload = json.loads(ledger.json_path.read_text(encoding="utf-8"))
    assert payload["task_order"] == ["en", "hi", "de"]
    assert payload["run_metadata"]["seed"] == 42
    assert payload["metrics"]["FWT"] == pytest.approx(0.125)
    assert len(payload["baselines"]) == 3
    assert len(payload["evaluations"]) == 12

    with ledger.csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert rows[0]["record_type"] == "baseline"
    assert rows[0]["split"] == "test"
    evaluation = next(row for row in rows if row["record_type"] == "evaluation")
    assert evaluation["trained_task"] == "en"
    assert evaluation["metadata_json"] == '{"accuracy_unit": "fraction"}'


def test_ledger_rejects_missing_data_before_computing_metrics(tmp_path):
    ledger = _make_ledger(tmp_path)
    with pytest.raises(ValueError, match="missing evaluations"):
        ledger.compute_metrics()


def test_ledger_rejects_duplicate_records_and_incomparable_splits(tmp_path):
    ledger = _make_ledger(tmp_path)
    ledger.record_evaluation(
        0, "en", 0.8, language="en", task="xnli", split="test",
        dataset="xnli", num_examples=100,
    )
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record_evaluation(
            0, "en", 0.81, language="en", task="xnli", split="test",
            dataset="xnli", num_examples=100,
        )
    with pytest.raises(ValueError, match="same language, task, split"):
        ledger.record_evaluation(
            1, "en", 0.75, language="en", task="xnli", split="validation",
            dataset="xnli", num_examples=100,
        )


def test_ledger_records_multiple_baseline_protocols(tmp_path):
    ledger = _make_ledger(tmp_path)
    _record_all(ledger)
    for task_id, pretrained_accuracy in zip(ledger.task_order, BASELINES):
        common = {
            "language": task_id,
            "task": "xnli",
            "split": "test",
            "dataset": "xnli",
            "num_examples": 100,
        }
        ledger.record_baseline(
            task_id,
            pretrained_accuracy + 0.1,
            baseline_name="single_task_finetuned",
            **common,
        )

    assert ledger.baseline_accuracies() == BASELINES
    assert ledger.baseline_accuracies("single_task_finetuned") == [
        value + 0.1 for value in BASELINES
    ]
    assert ledger.compute_metrics(
        "single_task_finetuned"
    )["FWT"] == pytest.approx(0.025)
    payload = json.loads(ledger.json_path.read_text(encoding="utf-8"))
    assert payload["zero_shot_baseline_name"] == "single_task_finetuned"
    assert payload["baseline_accuracies"] == [
        value + 0.1 for value in BASELINES
    ]


def test_ledger_validates_stage_accuracy_and_metadata(tmp_path):
    ledger = _make_ledger(tmp_path)
    with pytest.raises(ValueError, match="outside task_order"):
        ledger.record_evaluation(
            3, "en", 0.8, language="en", task="xnli", split="test",
            dataset="xnli", num_examples=100,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        ledger.record_baseline(
            "en", 1.5, language="en", task="xnli", split="test",
            dataset="xnli", num_examples=100,
        )
    with pytest.raises(TypeError, match="JSON-serializable"):
        ledger.record_baseline(
            "en", 0.5, language="en", task="xnli", split="test",
            dataset="xnli", num_examples=100, metadata={"bad": object()},
        )


def test_ledger_refuses_to_overwrite_an_existing_run(tmp_path):
    ledger = _make_ledger(tmp_path)
    ledger.record_baseline(
        "en", 0.5, language="en", task="xnli", split="test",
        dataset="xnli", num_examples=100,
    )

    with pytest.raises(FileExistsError, match="fresh output_dir"):
        _make_ledger(tmp_path)


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished_with = []

    def log(self, data, commit=True):
        self.logged.append((data, commit))

    def finish(self, exit_code=0):
        self.finished_with.append(exit_code)


class FakeWandbModule:
    def __init__(self):
        self.run = FakeWandbRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def test_wandb_logger_logs_stage_language_accuracy_and_summary(tmp_path):
    wandb = FakeWandbModule()
    logger = WandbRunLogger(
        "continual-learning",
        run_name="xnli-seed-42",
        config={"seed": 42, "model": "Llama-3.2-1B"},
        wandb_module=wandb,
    )
    ledger = EvaluationLedger(
        task_order=["en", "hi", "de"],
        experiment_id="xnli-seed-42",
        output_dir=tmp_path,
        run_metadata={"seed": 42},
        tracker=logger,
    )

    record = ledger.record_evaluation(
        1,
        "en",
        0.72,
        language="en",
        task="xnli",
        split="test",
        dataset="xnli",
        num_examples=100,
    )
    logger.log_metrics({"ACC": 0.71, "BWT": -0.08, "FWT": None})

    assert wandb.init_kwargs["project"] == "continual-learning"
    assert wandb.init_kwargs["name"] == "xnli-seed-42"
    assert "api_key" not in wandb.init_kwargs
    data, commit = wandb.run.logged[0]
    assert data["test_accuracy/en/xnli"] == record["accuracy"]
    assert data["stage_index"] == 1
    assert data["trained_task"] == "hi"
    assert commit is True
    assert wandb.run.logged[1][0] == {
        "final/ACC": 0.71,
        "final/BWT": -0.08,
    }
    assert wandb.run.summary == {
        "final/ACC": 0.71,
        "final/BWT": -0.08,
    }


def test_ledger_forwards_complete_stream_and_final_metrics_to_wandb(tmp_path):
    wandb = FakeWandbModule()
    logger = WandbRunLogger("continual-learning", wandb_module=wandb)
    ledger = _make_ledger(tmp_path, tracker=logger)
    _record_all(ledger)

    metrics = ledger.compute_metrics()

    assert len(wandb.run.logged) == 13
    assert wandb.run.logged[0][0][
        "baseline_accuracy/pretrained_zero_shot/test/en/xnli"
    ] == BASELINES[0]
    assert wandb.run.logged[3][0]["stage_index"] == 0
    assert wandb.run.logged[3][0]["test_accuracy/en/xnli"] == MATRIX[0][0]
    assert wandb.run.logged[6][0]["stage_index"] == 1
    assert wandb.run.logged[-1][0]["final/ACC"] == pytest.approx(
        metrics["ACC"]
    )
    assert wandb.run.summary["final/FWT"] == pytest.approx(metrics["FWT"])


def test_wandb_logger_logs_baselines_and_finishes_run():
    wandb = FakeWandbModule()
    with WandbRunLogger(
        "continual-learning",
        wandb_module=wandb,
    ) as logger:
        logger.log_record({
            "record_type": "baseline",
            "baseline_name": "pretrained_zero_shot",
            "language": "hi",
            "task": "xnli",
            "split": "test",
            "accuracy": 0.41,
            "num_examples": 100,
        })

    assert wandb.run.logged[0][0][
        "baseline_accuracy/pretrained_zero_shot/test/hi/xnli"
    ] == 0.41
    assert wandb.run.finished_with == [0]
    with pytest.raises(RuntimeError, match="finished"):
        logger.log_metrics({"ACC": 0.5})


def test_wandb_logger_validates_configuration_and_metrics():
    wandb = FakeWandbModule()
    with pytest.raises(ValueError, match="project"):
        WandbRunLogger("", wandb_module=wandb)

    logger = WandbRunLogger("continual-learning", wandb_module=wandb)
    with pytest.raises(ValueError, match="finite"):
        logger.log_metrics({"ACC": float("nan")})
