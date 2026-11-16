from types import SimpleNamespace

from datasets import Dataset
from peft import PeftModel
import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

from shared.continual_learning_lora import (
    DEFAULT_MODEL_NAME,
    LlamaLoRATrainer,
)


class TinyTokenizer:
    pad_token = None
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2
    padding_side = "right"

    def __call__(
        self,
        text,
        text_pair=None,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
    ):
        if isinstance(text, list):
            pairs = zip(text, text_pair or [None] * len(text))
            encoded = [
                self._encode(first, second, add_special_tokens, max_length)
                for first, second in pairs
            ]
            return {
                "input_ids": [row for row in encoded],
                "attention_mask": [[1] * len(row) for row in encoded],
            }
        ids = self._encode(text, text_pair, add_special_tokens, max_length)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    @staticmethod
    def _encode(first, second=None, add_special_tokens=True, max_length=None):
        ids = [3 + (ord(char) % 20) for char in first]
        if second is not None:
            ids += [4] + [3 + (ord(char) % 20) for char in second]
        if add_special_tokens:
            ids = [1] + ids
        if max_length is not None:
            ids = ids[:max_length]
        return ids or [1]

    def pad(self, features, *, padding=True, return_tensors=None, **kwargs):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            padding_length = max_length - len(feature["input_ids"])
            input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * padding_length
            )
            attention_masks.append(
                feature.get("attention_mask", [1] * len(feature["input_ids"]))
                + [0] * padding_length
            )
            if "labels" in feature:
                label = feature["labels"]
                if isinstance(label, list):
                    labels.append(label + [-100] * padding_length)
                else:
                    labels.append(label)
        batch = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
        }
        if labels:
            batch["labels"] = torch.tensor(labels)
        return batch

    def save_pretrained(self, output_dir):
        return (output_dir,)


def test_defaults_match_phase1_llama_configuration(tmp_path):
    wrapper = LlamaLoRATrainer(output_dir=tmp_path)

    assert wrapper.model_name == DEFAULT_MODEL_NAME
    assert wrapper.task_mode == "causal_lm"
    assert wrapper.lora_rank == 8
    assert wrapper.lora_alpha == 16
    assert wrapper.target_modules == (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"task_mode": "unknown"}, ValueError),
        ({"max_length": 1}, ValueError),
        ({"learning_rate": float("nan")}, ValueError),
        ({"lora_dropout": 1.0}, ValueError),
        ({"target_modules": ()}, ValueError),
        ({"num_labels": 1}, ValueError),
        ({"train_batch_size": True}, TypeError),
    ],
)
def test_constructor_rejects_invalid_settings(tmp_path, kwargs, exception):
    with pytest.raises(exception):
        LlamaLoRATrainer(output_dir=tmp_path, **kwargs)


def test_causal_prompt_target_masks_prompt_from_loss(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        task_mode="causal_lm",
        max_length=16,
    )
    wrapper.tokenizer = TinyTokenizer()
    dataset = Dataset.from_dict({
        "prompt": ["P"],
        "target": ["T"],
    })

    encoded = wrapper.prepare_dataset(dataset)[0]

    assert len(encoded["input_ids"]) <= 16
    assert encoded["labels"][:2] == [-100, -100]
    assert encoded["labels"][-2:] == [
        3 + ord("T") % 20,
        TinyTokenizer.eos_token_id,
    ]
    assert len(encoded["input_ids"]) == len(encoded["attention_mask"])
    assert len(encoded["input_ids"]) == len(encoded["labels"])


def test_causal_text_data_trains_on_all_tokens(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        task_mode="causal_lm",
        max_length=8,
    )
    wrapper.tokenizer = TinyTokenizer()
    encoded = wrapper.prepare_dataset(
        Dataset.from_dict({"text": ["hello"]})
    )[0]
    assert encoded["labels"] == encoded["input_ids"]


def test_xnli_classification_maps_string_labels(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        task_mode="sequence_classification",
    )
    wrapper.tokenizer = TinyTokenizer()
    dataset = Dataset.from_dict({
        "premise": ["a premise"],
        "hypothesis": ["a hypothesis"],
        "label": ["contradiction"],
    })

    encoded = wrapper.prepare_dataset(dataset)[0]

    assert encoded["labels"] == 2
    assert len(encoded["input_ids"]) == len(encoded["attention_mask"])


def test_binary_classification_can_use_custom_label_mapping(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        task_mode="sequence_classification",
        num_labels=2,
        label2id={"negative": 0, "positive": 1},
    )
    wrapper.tokenizer = TinyTokenizer()
    encoded = wrapper.prepare_dataset(
        Dataset.from_dict({"text": ["review"], "label": ["positive"]})
    )[0]
    assert encoded["labels"] == 1


def test_peft_adapter_attaches_to_tiny_llama_without_download(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        gradient_checkpointing=False,
    )
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    base_model = LlamaForCausalLM(config)
    model = wrapper.attach_lora(base_model, TinyTokenizer())

    assert isinstance(model, PeftModel)
    assert any(
        parameter.requires_grad for parameter in model.parameters()
    )
    inputs = torch.tensor([[1, 3, 5, 2]])
    output = model(input_ids=inputs, labels=inputs)
    output.loss.backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )


def test_model_loading_attaches_peft_lazily(tmp_path, monkeypatch):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        gradient_checkpointing=False,
    )
    tokenizer = TinyTokenizer()
    base_model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: base_model,
    )

    wrapper.load_model_and_tokenizer()

    assert wrapper.tokenizer is tokenizer
    assert isinstance(wrapper.model, PeftModel)


def test_sequence_classification_adapter_attaches_to_tiny_llama(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        task_mode="sequence_classification",
        num_labels=2,
        label2id={"negative": 0, "positive": 1},
        gradient_checkpointing=False,
    )
    model = LlamaForSequenceClassification(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            num_labels=2,
            pad_token_id=0,
        )
    )
    adapted = wrapper.attach_lora(model, TinyTokenizer())
    output = adapted(
        input_ids=torch.tensor([[1, 3, 5, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        labels=torch.tensor([1]),
    )
    output.loss.backward()

    assert isinstance(adapted, PeftModel)
    assert output.logits.shape == (1, 2)
    assert any("score.modules_to_save" in name for name, _ in adapted.named_parameters())


def test_sequential_stages_reuse_model_and_refuse_overwrite(
    tmp_path,
    monkeypatch,
):
    wrapper = LlamaLoRATrainer(output_dir=tmp_path)
    wrapper.model = object()
    wrapper.tokenizer = TinyTokenizer()
    calls = []

    class FakeTrainer:
        def train(self):
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def evaluate(self):
            return {"eval_loss": 0.4}

        def save_model(self, output_dir):
            return None

    def fake_make_trainer(**kwargs):
        calls.append(wrapper.model)
        return FakeTrainer()

    monkeypatch.setattr(wrapper, "_make_trainer", fake_make_trainer)

    first = wrapper.train_stage(Dataset.from_dict({"input_ids": [[1]], "labels": [[1]]}), "en")
    second = wrapper.train_stage(Dataset.from_dict({"input_ids": [[1]], "labels": [[1]]}), "hi")

    assert first["train_metrics"]["train_loss"] == 0.5
    assert second["stage_name"] == "hi"
    assert calls == [wrapper.model, wrapper.model]
    with pytest.raises(FileExistsError, match="already exists"):
        wrapper.train_stage(
            Dataset.from_dict({"input_ids": [[1]], "labels": [[1]]}),
            "en",
        )


def test_tiny_model_trains_and_saves_a_peft_stage(tmp_path):
    wrapper = LlamaLoRATrainer(
        output_dir=tmp_path,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        gradient_checkpointing=False,
        max_length=8,
    )
    tokenizer = TinyTokenizer()
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ))
    wrapper.tokenizer = tokenizer
    wrapper.model = wrapper.attach_lora(model, tokenizer)
    dataset = Dataset.from_dict({
        "input_ids": [[1, 3, 5, 2]],
        "attention_mask": [[1, 1, 1, 1]],
        "labels": [[-100, -100, 5, 2]],
    })

    result = wrapper.train_stage(dataset, "en")
    stage_path = tmp_path / "stage_en"

    assert result["train_metrics"]["train_loss"] >= 0
    assert (stage_path / "adapter_config.json").is_file()
    assert any(path.suffix == ".safetensors" for path in stage_path.iterdir())
