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
The published matrix file is not included in this repository; the paper page
currently exposes the matrix as a figure rather than a directly downloadable
numeric CSV. Do not digitize its colors and treat them as exact measurements.

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

The pilot metadata uses ISO 15924 script identifiers and broad language
families. It currently supports only the five Phase 2 pilot languages.

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

## Confirmatory B0 follow-up

The initial Kaggle RQ1 run found no Holm-significant prior, so do not select a
Phase 4 replay prior from this pilot alone. The result files are organized in
[`results/rq1_kaggle_2026-11-16/`](./results/rq1_kaggle_2026-11-16/README.md).

Use the [confirmatory B0 seeds notebook](./notebooks/phase3_confirmatory_b0_seeds.ipynb)
to run seeds 43 and 44 across the same five orders. It checks that the fixed
XNLI sample hash matches the original B0 run and exports ten validated result
records in a ZIP. After the run, combine those records with the original seed
42 B0 results and recompute RQ1 before proceeding to Phase 4.
