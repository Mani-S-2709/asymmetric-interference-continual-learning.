# Phase 1 results

## Matched stream-order comparison

The comparison protocol uses `facebook/xnli`, no replay, 1,000 training and
500 test examples per language, fixed test-selection seed 42, and training
seeds 42, 43, and 44. The two clean Colab templates use the same training code
and settings; language order and output folder are the intended differences.
Each language's training subset is selected with the same seed in both orders,
and both orders use the same fixed test subset.

| Stream order | Status | Notebook / result |
|---|---|---|
| English → Hindi → German | **Pending matched run.** Do not substitute the older 500/200 single-seed pilot. | [Clean Colab template](../notebooks/phase1_xnli_sequential_wandb_colab.ipynb) |
| Hindi → German → English | Complete for seeds 42, 43, and 44; the predefined checkpoint pattern passed 0/3 seeds. | [Completion summary](./xnli-hi-de-en-multiseed-train1000-test500-20261110T045241Z/kaggle_completion/README.md) |

Run the English → Hindi → German template to complete the matched comparison.
It writes to a separate Drive folder and creates a timestamped result group;
keep that group separate from the existing pilots. The complete comparison is
not available until this run has finished and its result records have been
validated.

## Separate English → Hindi → German pilots

- The compact [single-seed run](./xnli-en-hi-de-seed42-train500-test200-20261109T135235Z/evaluation/evaluation_results.json)
  uses 500 training and 200 test examples per language. Its mean final
  accuracy is 0.6183; it is **not** the matched 1,000/500 run.
- The legacy [baseline report](./pilot_baseline_report.md) and
  [metrics report](./pilot_continual_learning_metrics.md) describe a separate
  5,000-training / 1,000-test run. Those reports do not belong to either
  compact result group above.

Do not combine metrics across these different sample sizes, seeds, or
experiment groups. Source datasets, model checkpoints, full Kaggle archives,
and executed notebook outputs remain local and are not published.
