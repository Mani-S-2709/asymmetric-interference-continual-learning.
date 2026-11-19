"""Replay selection from a directed, signed interference matrix."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from shared.continual_learning_lora.replay import ReplayItem


class AsymmetricInterferenceReplaySelector:
    """Weight past-language replay by harm from the incoming language.

    The matrix uses ``scores[past_language][incoming_language]``. Negative
    scores indicate harm and are converted to risk with ``max(0, -score)``.
    """

    def __init__(
        self,
        scores: Mapping[str, Mapping[str, float]],
        *,
        seed: int = 0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self._scores = scores
        self._random = random.Random(seed)

    def language_weights(
        self,
        past_languages: Sequence[str],
        incoming_language: str,
    ) -> dict[str, float]:
        """Return normalized replay weights for the requested past languages."""
        if not isinstance(incoming_language, str) or not incoming_language.strip():
            raise ValueError("incoming_language must be a non-empty string.")

        if any(
            not isinstance(language, str) or not language.strip()
            for language in past_languages
        ):
            raise ValueError("past_languages must contain non-empty strings.")
        languages = sorted(set(past_languages))
        if not languages:
            return {}

        risks = {}
        for language in languages:
            row = self._scores.get(language)
            if row is None:
                raise ValueError(
                    f"No interference scores available for past language "
                    f"{language!r}."
                )
            if incoming_language not in row:
                raise ValueError(
                    f"Missing interference score for past language "
                    f"{language!r} and incoming language "
                    f"{incoming_language!r}."
                )

            score = row[incoming_language]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"Interference score for {language!r} and "
                    f"{incoming_language!r} must be a finite number."
                )
            try:
                score_value = float(score)
            except OverflowError as error:
                raise ValueError(
                    f"Interference score for {language!r} and "
                    f"{incoming_language!r} must be finite."
                ) from error
            if not math.isfinite(score_value):
                raise ValueError(
                    f"Interference score for {language!r} and "
                    f"{incoming_language!r} must be finite."
                )
            risks[language] = max(0.0, -score_value)

        maximum_risk = max(risks.values())
        if maximum_risk == 0:
            return {language: 1.0 / len(languages) for language in languages}

        scaled_risks = {
            language: risk / maximum_risk
            for language, risk in risks.items()
        }
        total_risk = sum(scaled_risks.values())
        return {
            language: risk / total_risk
            for language, risk in scaled_risks.items()
        }

    def select(
        self,
        candidates: Sequence[ReplayItem],
        budget: int,
        incoming_language: str,
    ) -> list[ReplayItem]:
        """Select up to ``budget`` distinct items, prioritizing risky languages."""
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an integer.")
        if budget < 0:
            raise ValueError("budget cannot be negative.")
        if budget == 0 or not candidates:
            return []

        items_by_language: dict[str, list[ReplayItem]] = defaultdict(list)
        for item in candidates:
            items_by_language[item.language].append(item)

        weights = self.language_weights(
            list(items_by_language),
            incoming_language,
        )
        remaining = {
            language: list(items)
            for language, items in items_by_language.items()
        }
        selected = []

        for _ in range(min(budget, len(candidates))):
            available_languages = [
                language for language in sorted(remaining) if remaining[language]
            ]
            available_weights = [
                weights[language] for language in available_languages
            ]
            if sum(available_weights) == 0:
                available_weights = [1.0] * len(available_languages)

            language = self._random.choices(
                available_languages,
                weights=available_weights,
                k=1,
            )[0]
            item_index = self._random.randrange(len(remaining[language]))
            selected.append(remaining[language].pop(item_index))

        return selected

    def select_with_replacement(
        self,
        candidates: Sequence[ReplayItem],
        sample_count: int,
        incoming_language: str,
    ) -> list[ReplayItem]:
        """Select a fixed-size replay batch, sampling examples with replacement."""
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise TypeError("sample_count must be an integer.")
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative.")
        if sample_count == 0 or not candidates:
            return []

        items_by_language: dict[str, list[ReplayItem]] = defaultdict(list)
        for item in candidates:
            items_by_language[item.language].append(item)
        weights = self.language_weights(
            list(items_by_language),
            incoming_language,
        )
        available_languages = sorted(items_by_language)
        selected = []
        for _ in range(sample_count):
            language = self._random.choices(
                available_languages,
                weights=[weights[language] for language in available_languages],
                k=1,
            )[0]
            selected.append(self._random.choice(items_by_language[language]))
        return selected
