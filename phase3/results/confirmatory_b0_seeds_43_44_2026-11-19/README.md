# Phase 3 confirmatory B0 follow-up — seeds 43 and 44

This folder contains the original seed-42 B0 records plus the downloaded
seed-43 and seed-44 export archives. It preserves 15 completed B0 result
records and recomputes the five-language RQ1 comparison against the verified
interference, URIEL+, and embedding priors.

## Contents

- `archives/`: unchanged seed-43/44 ZIP exports and archived copies of the
  corresponding clean notebooks.
- `results/seed_42/`, `results/seed_43/`, `results/seed_44/`: compact B0 result
  JSON records grouped by training seed.
- `combined_results/`: 15 result JSON records used as the analysis input.
- `manifests/` and `protocols/`: seed-43/44 protocol and prepared-data
  provenance.
- `priors/`: hash-verified matrices copied from the original RQ1 run.
- `analysis/`: combined forgetting observations, pair means, correlations,
  per-order correlations, and summary.
- `provenance/`: archive checksums and the organization record.

## Validation

Both confirmatory exports contain five completed orders and match the original
fixed-data manifest SHA-256:
`e459305688bf0ff5db4f744cee773ef2c0bb55c52df4bf71a7980479e58301a4`.

Across seeds 42–44, the mean B0 metrics over 15 orders are:

| Metric | Mean | Population SD |
|---|---:|---:|
| ACC | 0.5582 | 0.0403 |
| BWT | 0.0839 | 0.0434 |
| FWT | 0.1112 | 0.0242 |
| F | -0.0213 | 0.0604 |

## RQ1 result

No tested prior is Holm-significant. The strongest positive pooled correlations
are family mismatch (`rho=0.5841`, exact permutation `p=0.2000`,
Holm `p=1.0000`), URIEL+ syntactic distance (`rho=0.5555`, `p=0.0750`,
Holm `p=0.5250`), and embedding similarity (`rho=0.4936`, `p=0.1083`,
Holm `p=0.6500`). Grambank remains excluded because it lacks Spanish and
German coverage.

These results remain a five-language pilot across 15 seed/order runs. They
do not establish a winning replay prior for Phase 4.
