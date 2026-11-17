# Phase 1 results

## Matched stream-order comparison

The comparison protocol uses `facebook/xnli`, no replay, 1,000 training and
500 test examples per language, fixed test-selection seed 42, and training
seeds 42, 43, and 44. Both arms use the same training code and settings;
language order and output folder are the intended differences. Each language's
training subset is selected with the same seed in both orders, and both orders
use the same fixed test subset.

| Stream order | Status | Notebook / result |
|---|---|---|
| English → Hindi → German | Complete for seeds 42, 43, and 44; mean ACC 0.6911; checkpoint pattern passed 0/3 seeds. | [Run summary](./xnli-en-hi-de-multiseed-train1000-test500-20261116T074727501956Z/README.md); [Kaggle notebook](../notebooks/phase1_xnli_en_hi_de_multiseed_kaggle.ipynb) |
| Hindi → German → English | Complete for seeds 42, 43, and 44; mean ACC 0.7064; checkpoint pattern passed 0/3 seeds. | [Completion summary](./xnli-hi-de-en-multiseed-train1000-test500-20261110T045241Z/kaggle_completion/README.md) |

Both matched arms are complete. Mean ACC was 0.6911 for English → Hindi →
German and 0.7064 for Hindi → German → English, a descriptive difference of
-0.0153 (about -1.53 percentage points) across three seeds. This is not a
significance test or definitive evidence of a stream-order effect. The
predefined checkpoint pattern was not reproduced in either arm.

The compact English → Hindi → German result record contains the aggregate
summary and per-seed evaluation JSON/CSV files. The full Kaggle archive,
checkpoints, and executed notebook are kept locally under
`phase1/local_artifacts/` and are excluded from Git. Keep this run separate
from the older pilots below.

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
