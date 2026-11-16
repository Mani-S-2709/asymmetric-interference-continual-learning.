# B2 Kaggle run — 2026-11-11

Completed B2 uniform experience replay on Kaggle for all five frozen XNLI
stream orders, using seed 42. The uploaded Kaggle output archive was extracted
to this directory, and its result records were validated locally.

## Validation

- Protocol: `xnli-pilot-5lang-500x200-v1`
- Protocol SHA-256: `666044368518b0422705eab44e98a375650c60286ff8e560a7a24ab1944c0a3d`
- Data manifest SHA-256: `e459305688bf0ff5db4f744cee773ef2c0bb55c52df4bf71a7980479e58301a4`
- Completed and validated orders: 5 of 5 (`order-1` through `order-5`)
- Notebook execution errors: none; notebook reports successful validation of all five orders
- Result records: validated against their scheduled manifest entries
- Summary: `B2_summary.json` reports 5 completed of 5 expected orders

## Across-order results

| Metric | Mean | Population standard deviation |
|---|---:|---:|
| ACC | 0.6344 | 0.0143 |
| BWT | 0.0980 | 0.0275 |
| FWT | 0.1825 | 0.0299 |
| F | -0.0055 | 0.0172 |

Per-order metrics are in [`B2_summary.json`](./B2_summary.json); validated
records are in [`results/`](./results/), and the protocol and run manifest is
in [`protocol_manifest.json`](./protocol_manifest.json). Detailed evaluation
ledgers remain local and are not needed to validate or aggregate these compact
records.

The original 380 MB Kaggle archive, prepared XNLI sentence files, and model
checkpoints are retained locally and excluded from Git. The clean notebook
template is [`../../notebooks/phase2_b2_kaggle.ipynb`](../../notebooks/phase2_b2_kaggle.ipynb).
