# Phase 1 matched run: EN → HI → DE

Run group: `xnli-en-hi-de-multiseed-train1000-test500-20261116T074727501956Z`

This is the completed English → Hindi → German arm of the matched Phase 1
comparison. It uses `facebook/xnli`, sequential fine-tuning without replay,
1,000 training and 500 test examples per language, fixed test-selection seed
42, and training seeds 42, 43, and 44. The model is
`meta-llama/Llama-3.2-1B`; training ran on a Kaggle Tesla T4.

## Results

| Seed | ACC | BWT | FWT | Forgetting (F) | Checkpoint pattern |
|---:|---:|---:|---:|---:|:---:|
| 42 | 0.7093 | 0.1420 | 0.1880 | -0.0800 | No |
| 43 | 0.6787 | 0.1130 | 0.1780 | -0.1130 | No |
| 44 | 0.6853 | 0.1010 | 0.1730 | -0.0650 | No |
| Mean ACC | **0.6911** | — | — | — | **0/3 passed** |

Mean final accuracy by language is English 0.7440, Hindi 0.6507, and German
0.6787. These three seeds are a preliminary repeatability check, not a
statistical test or definitive reproduction.

## Contents

- `run_config.json` and `seed_progress.json` record the run configuration and
  completion state.
- `multiseed_summary.json` contains aggregate results and per-seed checks.
- `results/seed-{42,43,44}/` contains each seed's evaluation JSON and CSV.

The full Kaggle archive, checkpoints, and executed notebook are retained in
the local-only `phase1/local_artifacts/` directory and are excluded from Git.
The clean reusable Kaggle template remains at
[`phase1_xnli_en_hi_de_multiseed_kaggle.ipynb`](../../notebooks/phase1_xnli_en_hi_de_multiseed_kaggle.ipynb).
