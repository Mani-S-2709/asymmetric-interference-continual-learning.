# Phase 2 compact run records

This index maps the canonical current-protocol results and clean notebooks.
The repository includes compact validated result records, summaries, and
manifests. Kaggle working-directory archives, model checkpoints, tokenizers,
and fixed XNLI sentence data remain local because the archives are large and
contain artifacts that should not be mirrored here.

## Canonical results

The order-sensitive methods use five orders (seed 42). B1 is order-invariant
and has one run. Every selected result record validated against its matching
manifest. All nine methods use protocol
`xnli-pilot-5lang-500x200-v1`, protocol SHA-256
`666044368518b0422705eab44e98a375650c60286ff8e560a7a24ab1944c0a3d`, and
data-manifest SHA-256
`e459305688bf0ff5db4f744cee773ef2c0bb55c52df4bf71a7980479e58301a4`.

| Method | Canonical result source | Clean notebook | ACC mean (population SD) | BWT mean (SD) | FWT mean (SD) | Forgetting mean (SD) |
|---|---|---|---:|---:|---:|---:|
| B0 | [`b0_kaggle_2026-11-11/`](./b0_kaggle_2026-11-11/) | [`phase2_b0_kaggle.ipynb`](../notebooks/phase2_b0_kaggle.ipynb) | 0.5822 (0.0484) | 0.0983 (0.0447) | 0.1100 (0.0274) | -0.0458 (0.0475) |
| B1 | [`b1_joint_order_independent/`](./b1_joint_order_independent/) | [`phase2_b1_kaggle.ipynb`](../notebooks/phase2_b1_kaggle.ipynb) | 0.6230 (n/a) | n/a | n/a | n/a |
| B2 | [`b2_kaggle_2026-11-11/`](./b2_kaggle_2026-11-11/) | [`phase2_b2_kaggle.ipynb`](../notebooks/phase2_b2_kaggle.ipynb) | 0.6344 (0.0143) | 0.0980 (0.0275) | 0.1825 (0.0299) | -0.0055 (0.0172) |
| B3 | [`b3_kaggle_2026-11-11/`](./b3_kaggle_2026-11-11/) | [`phase2_b3_kaggle.ipynb`](../notebooks/phase2_b3_kaggle.ipynb) | 0.4206 (0.0079) | -0.0375 (0.0328) | 0.0283 (0.0145) | 0.0585 (0.0260) |
| B4 | [`b4_kaggle_2026-11-11/`](./b4_kaggle_2026-11-11/) | [`phase2_b4_kaggle.ipynb`](../notebooks/phase2_b4_kaggle.ipynb) | 0.6464 (0.0121) | 0.1183 (0.0285) | 0.1815 (0.0097) | -0.0250 (0.0180) |
| B5 | [`b5_kaggle_2026-11-11/`](./b5_kaggle_2026-11-11/) | [`phase2_b5_kaggle.ipynb`](../notebooks/phase2_b5_kaggle.ipynb) | 0.6020 (0.0271) | 0.0833 (0.0342) | 0.1558 (0.0144) | -0.0118 (0.0466) |
| B6 | [`b6_kaggle_2026-11-11/`](./b6_kaggle_2026-11-11/) | [`phase2_b6_kaggle.ipynb`](../notebooks/phase2_b6_kaggle.ipynb) | 0.5610 (0.0466) | 0.0943 (0.0382) | 0.1085 (0.0239) | -0.0313 (0.0276) |
| B7 | [`b7_kaggle_2026-11-11/`](./b7_kaggle_2026-11-11/) | [`phase2_b7_kaggle.ipynb`](../notebooks/phase2_b7_kaggle.ipynb) | 0.6138 (0.0090) | 0.0988 (0.0106) | 0.1548 (0.0089) | -0.0110 (0.0187) |
| B8 | [`b8_kaggle_2026-11-11/`](./b8_kaggle_2026-11-11/) | [`phase2_b8_kaggle.ipynb`](../notebooks/phase2_b8_kaggle.ipynb) | 0.6314 (0.0154) | 0.0865 (0.0127) | 0.1948 (0.0163) | 0.0010 (0.0116) |

B1's across-order SD is unavailable because its protocol has one
order-independent run. The SDs above are across the five stream orders, not
across independent training seeds. These are pilot estimates, not significance
tests. The v1 runs are not compute-matched: replay methods add replay exposures,
and B4 adds gradient-selection work. Treat those costs as part of the method
comparison and do not describe the v1 metric table as a matched-compute result.

## Repository contents

- Clean notebook templates are the `phase2_b*_kaggle.ipynb` files in
  [`../notebooks/`](../notebooks/). Use those to launch new work; do not edit an
  executed snapshot as though it were the template.
- Each canonical run directory contains the protocol manifest, method summary,
  and validated result JSONs. B3-B8 records were extracted from the complete
  Kaggle archives; model checkpoints, tokenizer files, prepared dataset text,
  and multi-hundred-megabyte archives are intentionally excluded from Git.
- Earlier and partial runs are not part of the canonical table. Do not combine
  them with the current-protocol results above.
- B3, B4, B5, B7, and B8 are documented algorithm adaptations, not claims of
  exact paper reproduction. B6 is also reported as an LR-ADJUST adaptation.

Detailed protocol deviations and the current Phase 2 status are documented in
[`../README.md`](../README.md).
