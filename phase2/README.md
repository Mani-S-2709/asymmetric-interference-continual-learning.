# Phase 2: B0-B8 baselines

Phase 2 pilot baseline coverage is complete for protocol
`xnli-pilot-5lang-500x200-v1`. The comparison uses English, Hindi, Spanish,
German, and Swahili; 500 training and 200 test examples per language; five
fixed orders for order-sensitive methods; and one order-independent B1 run.
All selected result records validate against their manifests and share data
manifest SHA-256
`e459305688bf0ff5db4f744cee773ef2c0bb55c52df4bf71a7980479e58301a4`.

The [run index](./runs/README.md) links compact summaries, manifests, result
records, and clean notebooks.

| Method | Canonical evidence | ACC mean (SD) | BWT mean (SD) | FWT mean (SD) | Forgetting mean (SD) |
|---|---|---:|---:|---:|---:|
| B0 | [`runs/b0_kaggle_2026-11-11/`](./runs/b0_kaggle_2026-11-11/) | 0.5822 (0.0484) | 0.0983 (0.0447) | 0.1100 (0.0274) | -0.0458 (0.0475) |
| B1 | [`runs/b1_joint_order_independent/`](./runs/b1_joint_order_independent/) | 0.6230 (n/a) | n/a | n/a | n/a |
| B2 | [`runs/b2_kaggle_2026-11-11/`](./runs/b2_kaggle_2026-11-11/) | 0.6344 (0.0143) | 0.0980 (0.0275) | 0.1825 (0.0299) | -0.0055 (0.0172) |
| B3 | [`runs/b3_kaggle_2026-11-11/`](./runs/b3_kaggle_2026-11-11/) | 0.4206 (0.0079) | -0.0375 (0.0328) | 0.0283 (0.0145) | 0.0585 (0.0260) |
| B4 | [`runs/b4_kaggle_2026-11-11/`](./runs/b4_kaggle_2026-11-11/) | 0.6464 (0.0121) | 0.1183 (0.0285) | 0.1815 (0.0097) | -0.0250 (0.0180) |
| B5 | [`runs/b5_kaggle_2026-11-11/`](./runs/b5_kaggle_2026-11-11/) | 0.6020 (0.0271) | 0.0833 (0.0342) | 0.1558 (0.0144) | -0.0118 (0.0466) |
| B6 | [`runs/b6_kaggle_2026-11-11/`](./runs/b6_kaggle_2026-11-11/) | 0.5610 (0.0466) | 0.0943 (0.0382) | 0.1085 (0.0239) | -0.0313 (0.0276) |
| B7 | [`runs/b7_kaggle_2026-11-11/`](./runs/b7_kaggle_2026-11-11/) | 0.6138 (0.0090) | 0.0988 (0.0106) | 0.1548 (0.0089) | -0.0110 (0.0187) |
| B8 | [`runs/b8_kaggle_2026-11-11/`](./runs/b8_kaggle_2026-11-11/) | 0.6314 (0.0154) | 0.0865 (0.0127) | 0.1948 (0.0163) | 0.0010 (0.0116) |

B1 has one order-independent run, so an across-order standard deviation does
not apply. Other standard deviations are across five orders with one training
seed; they are descriptive, not significance tests. The v1 methods are not
compute-matched: replay methods add example exposures, and B4 adds
gradient-selection work.

## Reproducibility and scope

The repository contains compact result records, summaries, protocol manifests,
source code, tests, and clean Kaggle templates. Fixed XNLI sentence files,
model checkpoints/tokenizers, executed notebook outputs, and the original
multi-hundred-megabyte Kaggle archives remain local. The manifests publish
selection indices and data checksums so the selected subsets can be recreated
and verified without redistributing the sentence text.

The local `artifacts/protocol_manifest.json` is a legacy manifest and must not
be used for new runs. Use a new protocol ID for any v2 rerun; do not overwrite
or aggregate it with these v1 results.

Several methods are documented adaptations, not exact source-paper
reproductions:

- B3 stores and distills logits for the three candidate answer labels rather
  than full-vocabulary DER++ logits.
- B4 is a GCR-style adaptation using CountSketch gradient summaries and
  gradient herding.
- B5 uses frozen-base prompt embeddings for nearest-neighbor replay.
- B6 uses a declared learning-rate decay factor and floor.
- B7 is a COMPASS-inspired semantic-cluster replay adaptation.
- B8 uses activation-anchor replay rather than the source paper's
  attention-head intervention and KL objective.

See the run index for the clean notebook templates and detailed per-method
records.
