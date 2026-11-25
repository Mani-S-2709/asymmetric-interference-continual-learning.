"""Load ordered language stages from Hugging Face dataset configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


@dataclass(frozen=True)
class StreamStage:
    """Data source and split for one language/task stage."""

    language: str
    task: str
    split: str = "train"
    dataset_id: str = "facebook/xnli"
    dataset_config: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("language", "task", "split", "dataset_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.dataset_config is not None and (
            not isinstance(self.dataset_config, str)
            or not self.dataset_config.strip()
        ):
            raise ValueError("dataset_config must be a non-empty string or None.")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be a non-empty string or None.")


class HuggingFaceStreamLoader:
    """Load each stage from its declared Hugging Face dataset source."""

    def __init__(
        self,
        stages: Sequence[StreamStage],
        batch_size: int = 8,
        max_samples_per_stage: int | None = None,
        shuffle_train: bool = True,
        seed: int = 42,
        dataset_loader: Callable[..., Any] | None = None,
    ) -> None:
        if not stages:
            raise ValueError("At least one stream stage must be provided.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_samples_per_stage is not None and max_samples_per_stage < 1:
            raise ValueError(
                "max_samples_per_stage must be at least 1 when specified."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        self.stages = tuple(stages)
        self.batch_size = batch_size
        self.max_samples_per_stage = max_samples_per_stage
        self.shuffle_train = shuffle_train
        self.seed = seed
        self.dataset_loader = dataset_loader

    def _dataset_loader(self) -> Callable[..., Any]:
        if self.dataset_loader is not None:
            return self.dataset_loader
        from datasets import load_dataset

        return load_dataset

    def load_stage(self, stage: StreamStage) -> Any:
        """Load a configured split, preserving each stage's source settings."""
        loader = self._dataset_loader()
        kwargs: dict[str, Any] = {"split": stage.split}
        if stage.revision is not None:
            kwargs["revision"] = stage.revision

        if stage.dataset_config is None:
            dataset = loader(stage.dataset_id, **kwargs)
        else:
            dataset = loader(stage.dataset_id, stage.dataset_config, **kwargs)

        if self.max_samples_per_stage is not None:
            sample_count = min(self.max_samples_per_stage, len(dataset))
            dataset = dataset.select(range(sample_count))
        return dataset

    def iter_stages(self) -> Iterator[tuple[StreamStage, Any]]:
        """Yield each stage and its batches in the declared stream order."""
        from torch.utils.data import DataLoader

        for stage in self.stages:
            dataset = self.load_stage(stage)
            if self.shuffle_train and stage.split == "train":
                dataset = dataset.shuffle(seed=self.seed)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
            yield stage, loader


class XNLIStreamLoader(HuggingFaceStreamLoader):
    """Backward-compatible loader name for existing XNLI callers."""
