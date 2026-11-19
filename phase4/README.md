# Phase 4: controlled ten-language pilot

Phase 4 is prepared but **training is gated** on the Phase 3 v2 decision
selecting the published-interference prior. Phase 3 v2 now uses this same
ten-language roster; its Arabic row and column use the pre-specified `k=5`
URIEL+ imputation described below. Do not edit the gate to `passed` unless the
corrected analysis and its imputation provenance have been reviewed and
recorded. If a different prior wins, stop and revise the Phase 4 method before
training.

## Frozen pilot

The protocol is
[`configs/controlled_10lang_pilot_protocol.json`](./configs/controlled_10lang_pilot_protocol.json).
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
the initial research-plan draft. The protocol freezes five distinct stream
orders and pairs them with seeds 42–46. B1 remains order-independent and runs
once per seed. B0–B8 plus M3 produce 50 records total.

This five-pair design is a **descriptive pilot**, not a significance-backed
headline experiment: five paired observations cannot resolve a Holm-corrected
test across nine baseline comparisons. The 5% figure refers to buffer
capacity; B2 and M3 use one replay example per current example. This does not
make every method compute-matched. The original Phase 4 hour estimate is
withdrawn until measured per-run time is available.

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

First complete and review Phase 3 v2 prior selection for this matching roster.
Then set both
`phase4.prior_gate.status` to `passed` and
`phase4.prior_gate.selected_prior` to `published_interference` in the frozen
protocol. This gate is checked before Phase 4 data preparation or training.

Prepare data and run one method/order pair at a time:

```powershell
python -m phase4.run_phase4 --method B0 --order-id order-1
python -m phase4.run_phase4 --method M3 --order-id order-1
```

The seed is selected from the frozen order/seed mapping. Repeat for all five
orders and methods B0–B8 plus M3. The first run prepares the fixed XNLI
subsets; later runs validate and reuse their manifest. Training requires a
Kaggle GPU (or compatible CUDA runtime), access to
`meta-llama/Llama-3.2-1B`, and an `HF_TOKEN` secret. The runner skips valid
completed result records and refuses mismatched protocol/data hashes.

After runs complete, aggregate the available records:

```powershell
python -m phase4.summarize_phase4
```

For separate Kaggle sessions, collect each output ZIP, extract the `results`
folders into one artifact root, and retain a `protocol_manifest.json` from
those runs. The summary command can aggregate using the saved checksummed data
metadata without requiring the raw XNLI examples. The report explicitly lists
incomplete method/run cells. It is descriptive only; do not report the pilot
as statistically significant.

## Kaggle notebook

[`notebooks/phase4_controlled_10lang_kaggle.ipynb`](./notebooks/phase4_controlled_10lang_kaggle.ipynb)
is a single-job template. Select a method and order in its configuration cell,
attach a Kaggle secret named `HF_TOKEN`, enable a GPU, and run only after the
gate is passed. The notebook clones this repository from the configured
`Mani-S-2709` origin. Push the prepared code to that repository before using
the notebook.

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
