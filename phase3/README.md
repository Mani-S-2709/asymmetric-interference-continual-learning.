# Phase 3: prior quality (RQ1)

This phase tests whether prior risk scores predict forgetting observed in the
completed Phase 2 B0 sequential runs. It analyzes saved accuracy trajectories;
**it does not train a model**. The embedding prior uses encoder inference and
can run on CPU; a GPU is optional.

## Important matrix convention

The published Interference Matrix paper defines
`I[affected_language, added_language] = (L_single - L_pair) / L_single`.
Rows are the language whose loss is evaluated; columns are the language added
to training. More-negative values mean more harm. The analysis converts the
published score to replay risk with `max(0, -I)`. This corrects the conflicting
sign and direction statements in the earlier research-plan draft.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m phase3.analyze_priors `
  --results-dir phase2\runs\b0_kaggle_2026-11-11\results `
  --output-dir phase3\results\rq1_pilot `
  --interference-matrix path\to\interference_matrix.csv `
  --uriel-distance path\to\uriel_syntactic_distance.csv `
  --embedding-similarity path\to\embedding_similarity.csv `
  --exclude-prior grambank_distance
```

The prior files are optional for a partial smoke test. For the current pilot,
Grambank is explicitly excluded due to language coverage; the remaining
planned inputs are the published interference matrix, URIEL+, and embeddings.
The numeric published matrix used by this project is retained at
[`results/rq1_kaggle_2026-11-16/priors/interference_matrix.csv`](./results/rq1_kaggle_2026-11-16/priors/interference_matrix.csv).
The paper page itself exposes the matrix as a figure rather than a directly
downloadable numeric CSV. Do not digitize its colors and treat them as exact
measurements.

## Prepare the prior matrices

The supplied URIEL+ CSV was reproduced with `urielplus==1.3.3`. To regenerate
it from the package:

```powershell
.\.venv\Scripts\python.exe -m pip install urielplus==1.3.3
.\.venv\Scripts\python.exe -m phase3.prepare_urielplus_distance `
  --output phase3\prior_inputs\uriel_syntactic_distance.csv
```

The script writes ISO-639-3 labels and a metadata sidecar with the package
version and CSV checksum. The analysis accepts those labels, ISO-639-1 labels,
and the paper's language identifiers.

The embedding prior is generated from the fixed Phase 2 XNLI training samples.
For each language, it embeds the premise/hypothesis pairs with
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, averages the
normalized vectors, and computes cosine similarity between language
centroids. It uses no labels and performs no fine-tuning. The script resolves
and records the model revision, data-manifest hash, sample counts, package
version, and CSV checksum:

```powershell
.\.venv\Scripts\python.exe -m pip install datasets sentence-transformers
.\.venv\Scripts\python.exe -m phase3.prepare_embedding_similarity `
  --protocol phase2\configs\pilot_protocol.json `
  --prepared-data-dir phase3\embedding_xnli_data `
  --output phase3\prior_inputs\embedding_similarity.csv
```

Grambank v1.0.3 does not contain Spanish or German records for this pilot. The
coverage checker writes `grambank_coverage.json` and refuses to create a
partial five-language matrix:

```powershell
.\.venv\Scripts\python.exe -m phase3.prepare_grambank_distance `
  --output-dir phase3\prior_inputs
```

This is a data-coverage limitation, not a missing download. Per the current
pilot design, Grambank is omitted and the coverage gap is recorded in
`grambank_coverage.json` and `summary.json`. Do not impute Spanish/German
values or substitute different languages silently. A future Grambank
comparison requires choosing a covered stream and rerunning the matching
Phase 2 baselines.

The published interference matrix remains an externally supplied input. Use
the user-verified file, named `interference_matrix.csv`, as a Kaggle input.

## CSV format for supplied matrices

Use one square wide CSV per matrix. The first column holds row language codes;
the remaining column headers use the same codes. For the published matrix,
rows must be affected/past languages and columns must be added/incoming
languages. Use the paper's language identifiers (for example `eng_Latn`,
`hin_Deva`, `spa_Latn`, `deu_Latn`, and `swh_Latn`) or the project ISO codes
`en`, `hi`, `es`, `de`, and `sw`. ISO 639-3 codes such as `eng`, `hin`, `spa`,
`deu`, and `swa` are also accepted. URIEL+ and Grambank files should contain
distances. The embedding file should contain similarities.

The metadata uses ISO 15924 script identifiers and broad language families.
It covers the Phase 2 pilot, Phase 3 ten-language protocol, and the approved
Phase 4 roster; it is not a general language metadata registry.

## Outputs and interpretation

The run writes:

- `forgetting_observations.csv`: each past-language accuracy change after an
  incoming language is added;
- `pair_forgetting.csv`: mean forgetting per directed language pair across
  orders;
- `prior_correlations.csv`: Spearman rank correlations, exact language-label
  permutation p-values for the five-language pilot, and Holm-adjusted p-values;
- `order_correlations.csv`: descriptive prior correlations for each stream order;
- `summary.json`: input gaps and interpretation notes.

The planned composites are included when their component files exist:
script + URIEL+ syntactic distance, and published interference + script. Each
component is converted to percentile ranks over all directed pilot-language
pairs, then equally averaged; weights are not fit to observed forgetting.
`summary.json` records any unavailable component or composite.

The permutation test permutes language labels across each prior matrix while
holding the observed pair-forgetting matrix fixed. It is exact for up to eight
languages (120 permutations for the five-language pilot); p-values across
available priors are Holm-adjusted. This tests whether the observed language
label alignment is unusual under label exchangeability. It does **not** measure
training-seed uncertainty, and five orders with one seed are not enough to
establish robustness. Use the per-order table and replicate with additional
training seeds before making a general claim. A separate reproducible
1,000-draw random-score control is also reported.

## Kaggle

Use [the Phase 3 Kaggle notebook](./notebooks/phase3_rq1_prior_comparison.ipynb).
It requires Kaggle Internet to clone the repository and, unless already
attached, fetch URIEL+, XNLI, and the embedding model. Encoder inference can
run on CPU; a Kaggle GPU is optional. Attach a dataset containing
`interference_matrix.csv`. The notebook generates URIEL+ and embedding
matrices, records the intentional Grambank exclusion and other missing inputs,
and exports a ZIP under `/kaggle/working`.

The repository changes must be pushed to the public repository before Kaggle
can clone them. A Hugging Face token is not needed for the public XNLI dataset
or encoder model; the notebook does download them if they are not cached.

## Expanded RQ1 v2 workflow

The original five-language pilot is complete and remains unchanged. Its
five-language label-permutation test has only 120 possible permutations; with
seven prior tests, Holm-adjusted significance at 0.05 is unattainable even at
the smallest possible raw permutation p-value. The v2 protocol therefore
expands the stream to ten XNLI languages before collecting new forgetting
observations:

`en, de, es, fr, sw, ar, ur, bg, ru, hi`

It keeps the 500/200 examples-per-language selection and B0 training settings,
uses three training seeds (`42`, `43`, `44`) and five frozen ten-language
orders, and uses a separate protocol ID. This is the same roster as the Phase 4
controlled stream. Two order pairs are reversals to
ensure all 90 directed language pairs are observed; the fifth order is another
fixed permutation. Do not mix v2 records with the five-language pilot.

Input preparation completed on 2026-11-20. The notebook had no cell errors,
and the generated ZIP validates against the current protocol. Its organized
copy remains a local run artifact for upload to Kaggle, not a repository file.
The next step is B0 training for seeds 42, 43, and 44; these runs and the
combined analysis are not yet complete.

The complete prepared workflow is:

1. Run
   [the v2 input-preparation notebook](./notebooks/phase3_rq1_v2_prepare_inputs.ipynb).
   It needs Kaggle Internet, but no GPU. It prepares the fixed XNLI selection,
   URIEL+ and embedding matrices, and a checksummed input ZIP. Because the
   roster and protocol ID were updated to match Phase 4, regenerate any
   previously prepared v2 input ZIP; training and analysis reject old-protocol
   bundles. Use the validated current bundle for this run.
2. Keep the resulting `phase3_rq1_v2_inputs.zip` in a **private Kaggle
   dataset**. It contains priors, protocol/provenance, and the fixed-data hash;
   it does not contain XNLI sentence text.
3. Run
   [the v2 B0 training notebook](./notebooks/phase3_rq1_v2_train_b0.ipynb)
   once for each of the 15 seed/order pairs (seeds 42–44 and orders 1–5).
   Each Kaggle session trains one order and produces its own result ZIP after
   completion. Attach the private input dataset each time, set both
   `SEED_TO_RUN` and `ORDER_TO_RUN`, and download each ZIP before starting the
   next run. The workflow checks the XNLI data hash before training.
4. Attach the input ZIP and each seed's complete result to
   [the v2 analysis notebook](./notebooks/phase3_rq1_v2_analyze.ipynb).
   For each seed, provide either its complete seed ZIP or all five per-order
   ZIPs. The workflow validates complete seed/order coverage, result records,
   and data hashes before computing pooled and per-order correlations.

The published interference source has scores for the nine roster languages
other than Arabic. The input-preparation notebook uses the Phase 4
pre-specified `k=5` URIEL+ syntactic-neighbor rule to impute Arabic's directed
row and column. The matrix and analysis provenance identify this as an
imputed—not directly published—prior and record the neighbor list and source
and output checksums. Grambank is excluded only if the generated coverage
report shows that it misses one or more roster languages; no values are
silently imputed.

**GPU requirement:** only B0 fine-tuning requires a Kaggle GPU/CUDA runtime.
The model is the gated `meta-llama/Llama-3.2-1B` checkpoint with the existing
4-bit LoRA settings, so accept its Hugging Face terms and provide a Kaggle
secret named `HF_TOKEN`. Input preparation and statistics run on CPU;
sentence-embedding inference can use a GPU for speed, but does not fine-tune.

The ten-language v2 roster matches Phase 4. The published-interference prior
uses the explicitly recorded Phase 4 Arabic imputation; do not describe the
result as a fully observed published matrix. Grambank coverage is checked
against all ten languages and any exclusion is recorded in the input summary.

With ten languages, the label test uses 10,000 Monte Carlo language-label
permutations and Holm correction. Three seeds are an initial expanded pilot,
not strong evidence of generalization or a guarantee of a winning prior.
Choose no winner unless its corrected test and consistency checks pass; do
not change the priors or analysis after seeing the results.

## Confirmatory B0 follow-up

The initial Kaggle RQ1 run found no Holm-significant prior, so do not select a
Phase 4 replay prior from this pilot alone. The result files are organized in
[`results/rq1_kaggle_2026-11-16/`](./results/rq1_kaggle_2026-11-16/README.md).

Seeds 43 and 44 have now been executed and exported separately using
[the seed 43 notebook](./notebooks/phase3_confirmatory_b0_seed_43.ipynb)
and [the seed 44 notebook](./notebooks/phase3_confirmatory_b0_seed_44.ipynb).
Each run covers the same five frozen orders and verifies the fixed XNLI data
hash. The unchanged downloads, grouped result records, combined 15-run RQ1
analysis, and provenance are organized in
[`results/confirmatory_b0_seeds_43_44_2026-11-19/`](./results/confirmatory_b0_seeds_43_44_2026-11-19/README.md).
