# Phase 2 B1 — completed joint multilingual run

This folder is the canonical compact result record for the current Phase 2 B1 run. It is intentionally separate from the older `b1_order1` run and from the original Kaggle download ZIP.

## Run summary

| Field | Value |
|---|---|
| Status | Completed and validated against the included manifest |
| Method | B1 joint multilingual training |
| Run ID | `xnli-pilot-5lang-500x200-v1-B1-order-independent-seed-42` |
| Order | `order-independent` (B1 is run once; it is not a sequential-order experiment) |
| Seed | 42 |
| Dataset | `facebook/xnli` |
| Languages | English (`en`), Hindi (`hi`), Spanish (`es`), German (`de`), Swahili (`sw`) |
| Training / test selection | 500 / 200 examples per language |
| Pooled training examples | 2,500 |
| Mean accuracy (ACC) | 0.623 |
| Per-language accuracy | `en`: 0.730, `hi`: 0.650, `es`: 0.620, `de`: 0.620, `sw`: 0.495 |
| Model | `meta-llama/Llama-3.2-1B` |
| LoRA | Rank 16, alpha 32, dropout 0.05; `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| W&B | Disabled for this run; the files here are the saved run artifacts |
| Source revision | `f38890fdce491fe3435e8d38fee89556711b030d` |

BWT, FWT, and forgetting are intentionally `null`: joint training has no sequential stage trajectory from which to calculate them. One run is the complete B1 protocol; do not compare it as five separate order runs.

## Contents

- `results/<run_id>.json`: completed result and run metadata.
- `protocol_manifest.json`: the exact protocol/run manifest used by the result.

The result's protocol hash is `666044368518b0422705eab44e98a375650c60286ff8e560a7a24ab1944c0a3d`; its data manifest hash is `e459305688bf0ff5db4f744cee773ef2c0bb55c52df4bf71a7980479e58301a4`. The result was validated against the `protocol_manifest.json` in this folder.

## Provenance and handling

The compact result and manifest were extracted from the local Kaggle working-directory archive. The fixed data and trained adapter remain local and are intentionally excluded from Git. The separate `../b1_order1/` result (ACC 0.671) is an older run under a different protocol and must not be merged with this B1 result.

For aggregation, use this folder's result JSON with this folder's manifest; do not substitute the stale `phase2/artifacts/protocol_manifest.json` or an older B1 result.
