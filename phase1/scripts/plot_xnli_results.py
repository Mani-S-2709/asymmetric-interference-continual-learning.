import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EXPERIMENT_ID = "xnli-en-hi-de-seed42-train500-test200-20261109T135235Z"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a local XNLI accuracy matrix.")
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = project_root / "phase1" / "results" / args.experiment_id
    csv_path = experiment_dir / "evaluation" / "evaluation_results.csv"

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        records = list(csv.DictReader(csv_file))

    evaluations = [record for record in records if record["record_type"] == "evaluation"]
    if not evaluations:
        raise ValueError(f"No evaluation records found in {csv_path}")

    stage_indices = sorted({int(record["stage_index"]) for record in evaluations})
    languages = list(dict.fromkeys(record["evaluated_task"] for record in evaluations))
    stage_tasks = {
        int(record["stage_index"]): record["trained_task"] for record in evaluations
    }
    values = np.full((len(stage_indices), len(languages)), np.nan)

    for record in evaluations:
        stage_position = stage_indices.index(int(record["stage_index"]))
        language_position = languages.index(record["evaluated_task"])
        values[stage_position, language_position] = float(record["accuracy"])

    if np.isnan(values).any():
        raise ValueError("The evaluation records do not form a complete stage-by-language matrix.")

    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(range(len(languages)), labels=languages)
    axis.set_yticks(
        range(len(stage_indices)),
        labels=[f"After {stage_tasks[index]}" for index in stage_indices],
    )
    axis.set_xlabel("Evaluation language")
    axis.set_ylabel("Training stage")
    axis.set_title("XNLI accuracy after each training stage")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")

    figure.colorbar(image, ax=axis, label="Accuracy")
    figure.tight_layout()
    output_path = experiment_dir / "accuracy_matrix.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved accuracy heatmap to: {output_path}")


if __name__ == "__main__":
    main()