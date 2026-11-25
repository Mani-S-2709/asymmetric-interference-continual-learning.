# Phase 3 RQ1 v2 input preparation — 2026-11-20

The Kaggle input-preparation notebook completed successfully for the revised
Phase 4 ten-language roster. Its input ZIP was validated against the current
frozen protocol and organized here.

## Validation summary

- Notebook execution count: 1; cell errors: 0.
- Protocol: `xnli-rq1-v2-phase4-roster-10lang-500x200-v2`.
- Languages: `en`, `de`, `es`, `fr`, `sw`, `ar`, `ur`, `bg`, `ru`, `hi`.
- Seeds: 42, 43, 44; orders: order-1 through order-5; expected B0 runs: 15.
- XNLI data-manifest SHA-256:
  `373852a7f1e09ab99da8a25f21023fedc1977ab1969e09d938874f7c645a721a`.
- The input ZIP passed the current workflow's protocol, roster, and prior
  checksum validation.
- Included priors: script mismatch, family mismatch, URIEL+ syntactic
  distance, embedding similarity, and their configured composites.
- Arabic interference scores use the pre-specified uniform mean of the five
  nearest languages in URIEL+ syntactic-distance space. The neighbors and
  source/output checksums are recorded in the input summary; these Arabic
  scores are imputed, not directly published measurements.
- Grambank is excluded because its coverage report lacks `de`, `es`, `ur`,
  and `bg`.

## Files

- `archives/phase3_rq1_v2_inputs.zip`: unchanged bundle to upload to a private
  Kaggle dataset.
- `notebooks/phase3_rq1_v2_prepare_inputs_executed.ipynb`: executed notebook
  copy.
- `provenance/`: the protocol, input summary, Grambank coverage report, and
  organization/checksum manifest.

## Next execution

Upload the ZIP from `archives/` as a private Kaggle dataset. Run
[`phase3_rq1_v2_train_b0.ipynb`](../../notebooks/phase3_rq1_v2_train_b0.ipynb)
15 times, once for each seed (`42`, `43`, `44`) and order (`order-1` through
`order-5`) pair. Set both `SEED_TO_RUN` and `ORDER_TO_RUN`, attach this private
input dataset each time, enable Kaggle GPU/CUDA, and retain every per-order
result ZIP. After the runs finish, attach one complete result per seed to
[`phase3_rq1_v2_analyze.ipynb`](../../notebooks/phase3_rq1_v2_analyze.ipynb):
either a complete seed ZIP or all five order ZIPs for that seed.
