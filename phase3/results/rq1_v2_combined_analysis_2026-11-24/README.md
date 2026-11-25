# Phase 3 RQ1 v2 combined analysis

- **Status:** Completed 2026-11-24
- **Protocol:** `xnli-rq1-v2-phase4-roster-10lang-500x200-v2`
- **Data-manifest SHA-256:** `373852a7f1e09ab99da8a25f21023fedc1977ab1969e09d938874f7c645a721a`

The executed Kaggle analysis validated all 15 B0 order results: five orders
for each of seeds 42, 43, and 44. It analyzed 90 directed language pairs and
675 forgetting observations using 10,000 language-label permutations with
Holm correction. No prior inputs were missing. Grambank was excluded because
its coverage does not include every configured language.

## Pooled prior correlations

| Prior | Spearman rho | One-sided p | Holm-adjusted p |
|---|---:|---:|---:|
| Script mismatch | -0.0358 | 0.6146 | 1.0000 |
| Family mismatch | 0.1235 | 0.3105 | 1.0000 |
| Published interference | -0.0150 | 0.5610 | 1.0000 |
| URIEL+ syntactic distance | 0.3325 | 0.0318 | 0.1908 |
| Embedding similarity | 0.4718 | 0.0065 | 0.0455 |
| Composite script + URIEL+ | 0.1636 | 0.1282 | 0.6409 |
| Composite interference + script | -0.0699 | 0.6896 | 1.0000 |

Embedding similarity is the only tested prior with a Holm-adjusted p-value
below 0.05 in this analysis. URIEL+ syntactic distance has a nominal
one-sided p-value below 0.05, but not after Holm correction.

## Interpretation limits

These permutation tests measure alignment with this pilot's observed
forgetting matrix; they do not capture training-seed uncertainty and are not
evidence of robustness across seeds. Treat the result as pilot evidence and
review the per-order correlations before making broader claims. The
random-score control used 10,000 repetitions.

## Artifacts

- [Analysis ZIP](./phase3_rq1_v2_analysis.zip) — contains the summary,
  provenance, pooled and per-order correlations, and forgetting observations.
- [Executed analysis notebook](./phase3_rq1_v2_analyze_executed.ipynb)
