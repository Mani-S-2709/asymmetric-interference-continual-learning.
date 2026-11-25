# Phase 3 RQ1 v2 per-order runs

Each seed/order folder contains the input ZIP to attach in Kaggle and the
associated notebook or run manifest. The input ZIPs are validated against
`xnli-rq1-v2-phase4-roster-10lang-500x200-v2`; their XNLI data-manifest and
prior-matrix checksums match.

## Order 1 status

- `seed-42/order-1/`: the executed notebook and training-result ZIP are saved
  in this folder. The notebook confirms seed 42/order 1 completed (1/1) with
  no notebook error. The result ZIP record confirms the same seed/order and
  its data-manifest hash matches the frozen protocol; see `run_manifest.json`
  for checksums and metrics.
- `seed-43/order-1/`: the provided notebook source is configured for seed 42,
  has no execution count, and its output only shows the seed-42 run starting.
  The later executed seed-43 notebook and result ZIP are now verified and
  stored in this folder. The original received notebook is preserved as-is.
- `seed-44/order-1/`: the provided notebook source and completed output both
  identify seed 42/order-1. The later executed seed-44 notebook and result ZIP
  are now verified and stored in this folder. The original received notebook
  is preserved as-is.

The original notebooks and input ZIPs are preserved. See each
`run_manifest.json` for checksums and the observed seed/order status. The
result ZIPs for seeds 43 and 44 are distinct from the shared input ZIP.

## Run orders 2 and 3

All six runs (seeds 42, 43, and 44; orders 2 and 3) are complete. Each run
folder contains its executed notebook, validated training-result ZIP, original
run notebook, shared input ZIP, and updated `run_manifest.json`. The notebook
outputs and result bundle metadata agree on seed/order; all result bundles
match the frozen v2 data-manifest hash. Per-run metrics are descriptive and
should be combined only in the v2 analysis workflow.

## Run orders 4 and 5

All six runs (seeds 42, 43, and 44; orders 4 and 5) are complete. Each run
folder contains its executed notebook, validated training-result ZIP, original
run notebook, shared input ZIP, and updated `run_manifest.json`. The notebook
outputs and result bundle metadata agree on seed/order; all result bundles
match the frozen v2 data-manifest hash. Per-run metrics are descriptive and
should be combined only in the v2 analysis workflow.

## Combined analysis

The combined analysis completed on 2026-11-24. It validated all 15 seed/order
results and the frozen data-manifest hash. See the [analysis report and
artifacts](../rq1_v2_combined_analysis_2026-11-24/README.md) for pooled and
per-order results, provenance, and interpretation limits.
