# Phase 1 — Hi → De → En three-seed completion

This folder contains the compact published results for experiment group
`xnli-hi-de-en-multiseed-train1000-test500-20261110T045241Z`.

## Status and results

Seed 44 completed, so all three requested seeds (42, 43, 44) now have validated
three-stage results for the Hindi → German → English stream. The experiment run
is complete, but the predefined checkpoint pattern was **not** reproduced:
the pass rate is 0/3. Do not describe this result as a successful reproduction
of the checkpoint pattern.

| Seed | ACC | BWT | FWT | Forgetting |
|---:|---:|---:|---:|---:|
| 42 | 0.7053 | 0.0640 | 0.2960 | -0.0140 |
| 43 | 0.7280 | 0.1180 | 0.2470 | -0.0380 |
| 44 | 0.6860 | 0.1230 | 0.1660 | -0.0440 |
| Mean ACC | 0.7064 | — | — | — |

Mean final accuracy by language: Hindi 0.6400, German 0.6887, English 0.7907.
Each seed uses 1,000 training and 500 fixed test examples per language, with
test-selection seed 42 and `meta-llama/Llama-3.2-1B`.

## Contents

- `multiseed_summary.json`: final three-seed aggregate and per-seed checks.
- `prior_results/seed-42-evaluation_results.json` and
  `prior_results/seed-43-evaluation_results.json`: validated results for the
  first two seeds.
- `seed-44-attempts/attempt-20261110T154618Z/evaluation/evaluation_results.json`:
  the successful seed-44 result.

Full Kaggle archives, model checkpoints, fixed dataset text, and executed
notebook outputs are retained locally and excluded from this repository. The
pre-completion summary and incomplete seed-44 attempt are not part of the
published result set.
