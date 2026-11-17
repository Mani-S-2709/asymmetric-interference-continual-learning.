# Phase 3 RQ1 Kaggle run — 2026-11-16

The downloaded result ZIPs and the three prior matrices used by the run are
organized here. The original ZIPs are preserved in `archives/`; the full
working-directory ZIP is not unpacked over the repository.

## Folder contents

- `analysis/`: forgetting tables, pooled/per-order correlations, and summary.
- `priors/`: interference, URIEL+, and embedding matrices used in the run.
- `provenance/`: run hashes, Grambank coverage/exclusion, and this run's file
  organization manifest.
- `archives/`: the original Kaggle result and full-working-directory ZIPs.

The interference and URIEL+ CSVs were copied from their local source files
only after their SHA-256 hashes matched the Kaggle run metadata. Grambank was
excluded because v1.0.3 lacks Spanish and German for this pilot.

## Pilot result

The run covers 5 B0 orders, 20 directed language pairs, and 50 forgetting
observations. The strongest raw correlations were family mismatch
(`rho=0.460`, exact one-sided `p=0.20`, Holm `p=1.00`), URIEL+ syntactic
distance (`rho=0.414`, `p=0.125`, Holm `p=0.875`), and embedding similarity
(`rho=0.345`, `p=0.15`, Holm `p=0.90`). No tested prior is significant after
Holm correction, so this pilot does not establish a winning prior.

The family-mismatch correlation is slightly above the random-score control's
97.5th percentile (`0.417`), but its exact label-permutation result is not
significant. Treat this as exploratory, not as confirmation.

## Recommended next step

Before selecting a prior for Phase 4, run the
[confirmatory B0 Kaggle notebook](../../notebooks/phase3_confirmatory_b0_seeds.ipynb).
It runs seeds 43 and 44 across the same five stream orders, checks the fixed
XNLI sample hash, and exports ten validated result records. Combine those
records with the original seed-42 B0 results and recompute RQ1. The current
pilot has only one training seed per order.
