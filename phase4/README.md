# Phase 4: controlled ten-language prior comparison

The supervisor-approved Phase 4 pilot is **complete: 15/15 runs** across B2,
B5, and M3, five frozen language orders, and training seed 42. The results are
a descriptive single-seed pilot and do not establish training-seed robustness.
Phase 3 did not select one winning prior: embedding similarity was the only
pooled prior with Holm-adjusted `p < 0.05` (Spearman `rho=0.4718`, adjusted
`p=0.0455`), while published interference was not selected. Accordingly,
Phase 4 compares three replay methods instead of representing one prior as
selected. See the
[Phase 3 combined-analysis report](../phase3/results/rq1_v2_combined_analysis_2026-11-24/README.md).

The [completed Phase 4 results report](./results/prior_comparison_seed42_2026-11-25/README.md)
records the checks, aggregated metrics, and limitations. The unified
[aggregate](./results/prior_comparison_seed42_2026-11-25/aggregate.json)
validates all 15 expected run records with no missing runs.

The approved design is B2 (uniform replay), B5 (embedding-similarity replay),
and M3 (published-interference replay), across all five frozen orders with
training seed 42: **15 runs**. Seed 42 is preselected, not selected from Phase 4
outcomes. This single-seed design is descriptive and cannot establish
seed-level robustness or support a significance claim. The approval gate
records the user's confirmation of supervisor approval; it does not claim
that Phase 3 selected `published_interference`.

## Approved comparison protocol

The protocol is
[`configs/controlled_10lang_prior_comparison_protocol.json`](./configs/controlled_10lang_prior_comparison_protocol.json).
It uses XNLI with 500 train and 200 test examples per language, a 250-example
buffer (5% of the 5,000 selected training examples), and these languages:

| XNLI code | Language | Script | Family |
|---|---|---|---|
| `en` | English | Latin | Indo-European |
| `de` | German | Latin | Indo-European |
| `es` | Spanish | Latin | Indo-European |
| `fr` | French | Latin | Indo-European |
| `sw` | Swahili | Latin | Niger-Congo |
| `ar` | Arabic | Arabic | Afro-Asiatic |
| `ur` | Urdu | Arabic | Indo-European |
| `bg` | Bulgarian | Cyrillic | Indo-European |
| `ru` | Russian | Cyrillic | Indo-European |
| `hi` | Hindi | Devanagari | Indo-European |

This is four scripts and three families, not the five scripts/four families in
the initial research-plan draft. The comparison schedules only B2, B5, and M3
on each order using seed 42. The earlier
[`controlled_10lang_pilot_protocol.json`](./configs/controlled_10lang_pilot_protocol.json)
is preserved unchanged as the original, now superseded protocol; do not use it
for this approved comparison.

This 15-run design is a **descriptive pilot**, not a significance-backed
headline experiment: one seed cannot establish training-seed robustness. The
5% figure refers to buffer
capacity; B2 and M3 use one replay example per current example. This does not
make every method compute-matched. The completed Kaggle notebooks record
8.90–10.34 hours per order (mean 9.49 hours) for the three-method run cell;
actual runtime varies with Kaggle hardware and scheduling.

## Arabic interference-score imputation

The checked-in published matrix covers the other nine target languages but
has no Arabic row or column. The approved protocol uses the plan's URIEL+
k-nearest-neighbor approach with `k=5`: select the five covered languages
nearest to Arabic in URIEL+ syntactic-distance space, then take an unweighted
mean of their directed rows and columns. The Arabic diagonal is zero. The
prepared matrix records the neighbors, distances, source-matrix checksum,
package version, and output checksum; loading it verifies its checksum and
coverage. Unsupported or missing URIEL+ distances fail instead of being
silently replaced.

Prepare the matrix from the repository root in an environment with the Phase 2
dependencies and `urielplus==1.3.3` installed:

```powershell
python -m phase4.prepare_interference_matrix
```

The generated files go under `phase4\artifacts\` and are intentionally not
source-controlled experiment results.

## Running and aggregating

The comparison approval gate is already recorded in the new protocol. Keep
the original protocol and its pending single-prior gate unchanged. Prepare data
and run one method/order/seed combination at a time:

```powershell
python -m phase4.run_phase4 --method B2 --order-id order-1 --seed 42
python -m phase4.run_phase4 --method B5 --order-id order-1 --seed 42
python -m phase4.run_phase4 --method M3 --order-id order-1 --seed 42
```

Repeat these three methods for orders 1–5. The first run prepares the fixed XNLI
subsets; later runs validate and reuse their manifest. Training requires a
Kaggle GPU (or compatible CUDA runtime), access to
`meta-llama/Llama-3.2-1B`, and an `HF_TOKEN` secret. The runner skips valid
completed result records and refuses mismatched protocol/data hashes.

## Approved single-seed extension: remaining baselines

The supervisor-approved extension runs the remaining methods at seed 42 on
the same five frozen orders and data settings. It uses a separate
[extension protocol](./configs/controlled_10lang_full_methods_seed42_extension_protocol.json)
so the original B2/B5/M3 protocol and its 15 result records remain unchanged.
B0, B3, B4, B6, B7, and B8 each have five order-specific runs. B1 is joint and
order-independent, so it has one run. The extension therefore expects 31
records; together with the original 15, the screening comparison will contain
46 records validated under two protocol IDs. Preserve both protocol hashes in
any combined report.

### Current extension status: B1 completed

The order-independent B1 seed-42 run is validated with ACC 0.693. It is the
only completed extension record so far: the extension aggregate is therefore
incomplete at 1/31, with 30 records still missing. Do not treat it as the final
full-baseline comparison. The executed notebook, output archive, extracted
record, and validation details are in the
[B1 execution folder](./results/full_methods_seed42_b1_joint_2026-11-25/README.md).

### Kaggle notebooks

Attach
[`phase4_full_baselines_code.zip`](./artifacts/phase4_full_baselines_code.zip),
enable a GPU and Internet, accept the gated model terms, and configure the
`HF_TOKEN` Kaggle secret. Attach an earlier output ZIP with the same batch ID
only when resuming an interrupted notebook.

- [B1 joint run](./notebooks/phase4_full_methods_seed42_b1_joint.ipynb)
- B0/B3/B4:
  [order 1](./notebooks/phase4_full_methods_seed42_order-1_b0_b3_b4.ipynb),
  [order 2](./notebooks/phase4_full_methods_seed42_order-2_b0_b3_b4.ipynb),
  [order 3](./notebooks/phase4_full_methods_seed42_order-3_b0_b3_b4.ipynb),
  [order 4](./notebooks/phase4_full_methods_seed42_order-4_b0_b3_b4.ipynb),
  [order 5](./notebooks/phase4_full_methods_seed42_order-5_b0_b3_b4.ipynb)
- B6/B7/B8:
  [order 1](./notebooks/phase4_full_methods_seed42_order-1_b6_b7_b8.ipynb),
  [order 2](./notebooks/phase4_full_methods_seed42_order-2_b6_b7_b8.ipynb),
  [order 3](./notebooks/phase4_full_methods_seed42_order-3_b6_b7_b8.ipynb),
  [order 4](./notebooks/phase4_full_methods_seed42_order-4_b6_b7_b8.ipynb),
  [order 5](./notebooks/phase4_full_methods_seed42_order-5_b6_b7_b8.ipynb)

Each notebook saves a checkpoint ZIP after each method cell. Save/download its
latest Kaggle output before ending the session; to resume, attach that output
ZIP as a Kaggle input to the same notebook. A run is skipped only after its
record validates against the extension protocol and prepared-data hash.

### GPU-time estimates and observations

The completed B2/B5/M3 notebooks took 8.90–10.34 hours per order (mean
9.49 hours) on the recorded Kaggle hardware. The order-specific extension
groups have not yet been benchmarked; their times are planning estimates, not
guarantees. B1 now has one observed training-stage runtime:

| Notebook workload | GPU time |
|---|---:|
| B0 + B3 + B4, one order | 9–13 hours; B4 may increase this |
| B6 + B7 + B8, one order | 8–11 hours |
| B1, one joint run | 30m 31s training stage on a Tesla T4; total notebook time is longer |

The B1 time is the recorded Trainer `train_runtime` for one run, not a
hardware-independent guarantee or the full notebook elapsed time; data/model
downloads, setup, and evaluation add time. The other group estimates are rough
extrapolations from the completed three-method runs. Check the current Kaggle
session limit and GPU quota before each run. If a grouped notebook may exceed
the available session time, its separate method cells and checkpoints allow
resuming without retraining completed methods.

To reproduce the aggregate from the archived result records:

```powershell
python -m phase4.summarize_phase4 `
  --protocol phase4\configs\controlled_10lang_prior_comparison_protocol.json `
  --artifact-root phase4\results\prior_comparison_seed42_2026-11-25
```

The [archived result folder](./results/prior_comparison_seed42_2026-11-25/)
contains all five original output ZIPs, the unified manifest, 15 result
records, shared M3 matrix files, provenance checks, and aggregate. The
summary command uses the saved checksummed data metadata and does not require
raw XNLI examples. Report the comparison as descriptive only; do not claim
statistical significance or seed-level robustness.

## Kaggle notebook

Order-specific Kaggle notebook templates (the corresponding submitted
executions were checked and validated) are:

- [Order 1](./notebooks/phase4_controlled_10lang_kaggle_order1.ipynb)
- [Order 2](./notebooks/phase4_controlled_10lang_kaggle_order2.ipynb)
- [Order 3](./notebooks/phase4_controlled_10lang_kaggle_order3.ipynb)
- [Order 4](./notebooks/phase4_controlled_10lang_kaggle_order4.ipynb)
- [Order 5](./notebooks/phase4_controlled_10lang_kaggle_order5.ipynb)

The [general notebook template](./notebooks/phase4_controlled_10lang_kaggle.ipynb)
can also be configured for one order at a time. Attach a Kaggle secret named
`HF_TOKEN`, enable Internet and a GPU, and accept the gated model terms. Attach
[`artifacts/phase4_prior_comparison_code.zip`](./artifacts/phase4_prior_comparison_code.zip)
as a Kaggle input dataset. The notebook reads the bundled source directly and
does not require GitHub cloning or authentication. In the bundle, the runner
records a deterministic source-tree SHA-256 fingerprint because no `.git`
metadata is present; Git checkouts continue to record the commit hash.

## Implementation and tests

- [`run_phase4.py`](./run_phase4.py) delegates B0–B8 to the Phase 2 runner and
  implements M3 with the same fixed-size, with-replacement replay exposure as
  B2.
- [`interference_matrix.py`](./interference_matrix.py) validates and records
  the directional matrix and the Arabic imputation.
- [`interference_replay.py`](./interference_replay.py) weights past-language
  replay by the negative directed interference score.

Run focused tests from the repository root:

```powershell
python -m pytest phase2\tests\test_protocol.py `
  phase2\tests\test_run_baseline.py::test_run_baseline_skips_a_valid_completed_run_without_gpu `
  phase2\tests\test_run_baseline.py::test_run_baseline_selects_one_paired_order_and_seed_without_gpu `
  phase3\tests\test_prepare_priors.py phase4\tests `
  shared\tests\test_continual_learning_replay.py
```
