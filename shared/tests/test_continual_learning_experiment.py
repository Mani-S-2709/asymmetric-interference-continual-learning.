import json
from types import SimpleNamespace

import pytest
import torch

from shared.continual_learning_lora import (
    evaluate_xnli_accuracy,
    run_sequential_experiment,
)
from shared.continual_learning_metrics import WandbRunLogger


class CandidateTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        if text.startswith(" "):
            ids = {
                " entailment": [4],
                " neutral": [5],
                " contradiction": [6],
            }[text]
        else:
            ids = [3, 3, 3]
        if add_special_tokens:
            ids = [1] + ids
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class CandidateScoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, logits_to_keep=0):
        logits = torch.zeros(
            (*input_ids.shape, 8),
            device=input_ids.device,
        )
        logits[..., 4] = 4
        logits[..., 5] = 2
        logits[..., 6] = 1
        logits[..., 2] = 2
        if isinstance(logits_to_keep, int):
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
        else:
            logits = logits[:, logits_to_keep, :]
        return SimpleNamespace(logits=logits)


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = []

    def log(self, data, commit=True):
        self.logged.append(data)

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FakeWandbModule:
    def __init__(self):
        self.run = FakeWandbRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


class FakeSequentialTrainer:
    model_name = "tiny-local-llama"
    seed = 7

    def __init__(self):
        self.model = object()
        self.tokenizer = object()
        self.stage = 0
        self.trained = []

    def train_stage(self, dataset, stage_name):
        self.trained.append(stage_name)
        self.stage += 1
        return {"output_dir": f"stage_{stage_name}"}


def test_xnli_evaluator_scores_labels_and_restores_model_mode():
    model = CandidateScoringModel()
    model.train()
    dataset = [
        {
            "premise": "premise one",
            "hypothesis": "hypothesis one",
            "label": 0,
        },
        {
            "premise": {"en": "premise two", "hi": "p2"},
            "hypothesis": {
                "language": ["en", "hi"],
                "translation": ["hypothesis two", "h2"],
            },
            "label": 2,
        },
    ]

    accuracy = evaluate_xnli_accuracy(
        model,
        CandidateTokenizer(),
        dataset,
        "en",
        batch_size=1,
    )

    assert accuracy == 0.5
    assert model.training is True


def test_xnli_evaluator_rejects_missing_language_translation():
    with pytest.raises(ValueError, match="premise translation"):
        evaluate_xnli_accuracy(
            CandidateScoringModel(),
            CandidateTokenizer(),
            [{
                "premise": {"language": {"en": "English"}},
                "hypothesis": {"language": ["en"], "translation": ["text"]},
                "label": 0,
            }],
            "hi",
        )


def test_sequential_runner_logs_baselines_and_every_language_after_each_stage(
    tmp_path,
):
    wandb_module = FakeWandbModule()
    tracker = WandbRunLogger(
        "continual-learning",
        run_name="tiny-sequential",
        wandb_module=wandb_module,
    )
    trainer = FakeSequentialTrainer()
    train_datasets = {"en": ["en train"], "hi": ["hi train"]}
    test_datasets = {"en": [1, 2], "hi": [1, 2, 3]}
    results_by_stage = {
        0: {"en": 0.70, "hi": 0.40},
        1: {"en": 0.80, "hi": 0.45},
        2: {"en": 0.65, "hi": 0.75},
    }

    def evaluator(model, tokenizer, dataset, language):
        assert model is trainer.model
        assert tokenizer is trainer.tokenizer
        return results_by_stage[trainer.stage][language]

    result = run_sequential_experiment(
        trainer,
        task_order=["en", "hi"],
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        experiment_id="tiny-sequential",
        output_dir=tmp_path / "ledger",
        evaluator=evaluator,
        run_metadata={"phase": 1},
        dataset_name="xnli",
        task_name="xnli",
        tracker=tracker,
    )

    assert trainer.trained == ["00_en", "01_hi"]
    assert [stage["test_accuracy"] for stage in result["stages"]] == [
        results_by_stage[1],
        results_by_stage[2],
    ]
    assert result["metrics"]["ACC"] == pytest.approx(0.7)
    assert result["metrics"]["BWT"] == pytest.approx(-0.15)
    assert result["metrics"]["FWT"] == pytest.approx(0.05)

    payload = json.loads(result["ledger"].json_path.read_text(encoding="utf-8"))
    assert payload["baseline_accuracies"] == [0.70, 0.40]
    assert payload["performance_matrix"] == [
        [0.80, 0.45],
        [0.65, 0.75],
    ]
    assert len(wandb_module.run.logged) == 2 + 2 * 2 + 1
    assert wandb_module.run.logged[2]["stage_index"] == 0
    assert wandb_module.run.logged[2]["trained_task"] == "en"
    assert (
        wandb_module.run.logged[2]["test_accuracy/en/xnli"] == 0.80
    )
    assert wandb_module.run.summary["final/ACC"] == pytest.approx(0.7)


def test_runner_rejects_incomplete_task_datasets_before_training(tmp_path):
    trainer = FakeSequentialTrainer()
    with pytest.raises(ValueError, match="exactly match task_order"):
        run_sequential_experiment(
            trainer,
            task_order=["en", "hi"],
            train_datasets={"en": []},
            test_datasets={"en": [1], "hi": [1]},
            experiment_id="invalid",
            output_dir=tmp_path,
        )
    assert trainer.trained == []


def test_runner_finishes_its_wandb_run_when_evaluation_fails(
    tmp_path,
    monkeypatch,
):
    wandb_module = FakeWandbModule()
    trainer = FakeSequentialTrainer()
    from shared.continual_learning_lora import experiment

    def fake_wandb_logger(**kwargs):
        return WandbRunLogger(
            kwargs["project"],
            run_name=kwargs["run_name"],
            wandb_module=wandb_module,
        )

    monkeypatch.setattr(experiment, "WandbRunLogger", fake_wandb_logger)

    def failing_evaluator(*args):
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_sequential_experiment(
            trainer,
            task_order=["en"],
            train_datasets={"en": ["train"]},
            test_datasets={"en": ["test"]},
            experiment_id="failing-run",
            output_dir=tmp_path / "ledger",
            evaluator=failing_evaluator,
            wandb_project="continual-learning",
        )
    assert wandb_module.run.finished == [1]
