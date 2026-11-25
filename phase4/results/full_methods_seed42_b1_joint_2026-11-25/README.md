# Phase 4 B1 joint-run record

**Status: validated — 1 of 31 extension records complete.**

This is the seed-42, order-independent B1 joint-training run from the approved
full-baseline extension. It is a valid single B1 result, not a completed
extension comparison. The bundled aggregate is expected to be incomplete:
30 scheduled extension runs are still missing.

## Run and result

| Field | Value |
|---|---|
| Run ID | `xnli-phase4-full-methods-10lang-500x200-seed42-extension-v1-B1-order-independent-seed-42` |
| Method | B1 joint multilingual training |
| Order | `order-independent` |
| Training seed | 42 |
| Dataset | XNLI; 500 train and 200 test examples for each of 10 languages |
| Pooled training examples | 5,000 |
| Model | `meta-llama/Llama-3.2-1B` |
| Evaluation mode | Joint |
| ACC / AA / FP | 0.693 |
| GPU | Tesla T4 |
| Trainer training runtime | 1,830.8086 seconds (about 30m 31s) |

Final accuracy by language:

| Language | ACC |
|---|---:|
| English (`en`) | 0.775 |
| German (`de`) | 0.685 |
| Spanish (`es`) | 0.735 |
| French (`fr`) | 0.775 |
| Swahili (`sw`) | 0.515 |
| Arabic (`ar`) | 0.705 |
| Urdu (`ur`) | 0.600 |
| Bulgarian (`bg`) | 0.705 |
| Russian (`ru`) | 0.745 |
| Hindi (`hi`) | 0.690 |

BWT, FWT, and forgetting are not defined for this joint-training run and are
`null` in the record. B1 is scheduled once as order-independent; it should not
be repeated for each of the five stream orders.

## Accuracy comparison CSV

[`accuracy_comparison_B1_B2_B5_M3.csv`](./accuracy_comparison_B1_B2_B5_M3.csv)
compares B1 with the completed B2/B5/M3 pilot overall and by language. The
overall row reports B1 ACC 0.693, compared descriptively with the five-order
means B2 0.6807, B5 0.6632, and M3 0.6858. B1 is one joint run; each existing
baseline value is a mean across five orders at seed 42. The SD columns describe
variation across those five orders, not training seeds. The deltas are
descriptive differences, not significance tests or a confirmatory ranking.

## Validation and provenance

- Every code cell in the executed notebook completed; no notebook execution
  errors were recorded.
- Executed code-cell sources match the prepared B1 extension notebook.
- The output ZIP passed its integrity check, and its result record matches the
  frozen extension protocol and prepared-data manifest hashes. The archive
  does not contain raw prepared XNLI files, so their contents were not
  independently rehashed from this output.
- The extracted `aggregate.json` reports 1/31 completed runs. It is a progress
  snapshot, not the final extension aggregate.
- Protocol SHA-256:
  `842767e85c7649123639ee3a760d833387958a7e40ca8357610e1b81071f0f46`
- Prepared-data manifest SHA-256:
  `373852a7f1e09ab99da8a25f21023fedc1977ab1969e09d938874f7c645a721a`
- Source-tree SHA-256:
  `3d292574867bba545ad8bcea28513c51a1a1b35a35146043aa7f108cc4565167`
- Executed notebook SHA-256:
  `bb373ccc9727e029cf2800fa8d1537d022b2d3ce8518ddcc798aed8bc4d5c552`
- Kaggle output ZIP SHA-256:
  `987676e7cab2b2bb21b40e6606166a106cf92d536d6d3fd1c6d2e193ccd3c8ce`

The training runtime is the Trainer-reported training-stage time, not total
Kaggle session time. Setup, dataset/model downloads, and evaluation add time.

## Files

- [`notebooks/phase4_full_methods_seed42_b1_joint_executed.ipynb`](./notebooks/phase4_full_methods_seed42_b1_joint_executed.ipynb):
  executed notebook supplied for this run.
- [`input_archives/phase4_full_methods_seed42_b1_joint.zip`](./input_archives/phase4_full_methods_seed42_b1_joint.zip):
  original Kaggle output archive.
- [`protocol_manifest.json`](./protocol_manifest.json): frozen protocol and
  prepared-data hashes.
- [`results/`](./results/): validated B1 result record.
- [`aggregate.json`](./aggregate.json): partial 1/31 extension summary.
- [`accuracy_comparison_B1_B2_B5_M3.csv`](./accuracy_comparison_B1_B2_B5_M3.csv):
  descriptive overall and per-language ACC comparison.
- [Prepared B1 notebook template](../../notebooks/phase4_full_methods_seed42_b1_joint.ipynb)
  and [extension protocol](../../configs/controlled_10lang_full_methods_seed42_extension_protocol.json).

The original B2/B5/M3 records remain under their original protocol and are not
modified by this B1 result.
