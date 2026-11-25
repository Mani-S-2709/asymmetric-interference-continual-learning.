# Project status

**Last reviewed:** 2026-11-25
**Current focus:** Run and validate the supervisor-approved Phase 4 single-seed extension for the remaining baselines, then review the complete descriptive screen before proposing a separately frozen multi-seed follow-up.

## At a glance

| Workstream | Status | What is established |
|---|---|---|
| Phase 0: research onboarding | **Complete: paper/concept checklists and archived experiment results recorded** | The eight paper readings and concepts checklists are complete per the researcher's report (reading notes remain unfilled). The archived English→Hindi run records English ACC 0.914 before and 0.870 after Hindi training (4.4 percentage-point forgetting) and Hindi ACC 0.8939. Data, results, checkpoints, report, and source snapshots are extracted locally; they remain Git-ignored. A local rerun attempt was blocked because PyTorch is missing from the current `.venv`. See the [Phase 0 record](./phase0/README.md). |
| Phase 1: harness and pilot evidence | Matched stream-order comparison complete; target checkpoint pattern not reproduced; mixed-source EN → HI → BN runner prepared, not run | Both 1,000/500 EN → HI → DE and HI → DE → EN arms are complete for seeds 42–44; each passed the predefined checkpoint pattern in 0/3 seeds. A reusable source-aware stream loader and pinned EN → HI → BN IndicXNLI config are prepared but not executed; verify the IndicXNLI license before using or publishing results. Keep this new stream separate from prior runs. See the [Phase 1 workflow](./phase1/README.md). |
| Phase 2: B0–B8 baselines | Pilot baseline coverage complete; source-paper reproductions remain qualified | B0, B2–B8 each have five current-protocol orders; B1 has its single order-independent run. Every selected result record validates against its matching manifest. B3 and B7 (and other described methods) are adaptations, not exact paper reproductions. |
| Phase 3: prior comparison (RQ1) | Original pilot and seed-43/44 follow-up complete; v2 inputs, all 15 B0 seed/order runs, and combined analysis complete | V2 matches the Phase 4 roster (`en`, `de`, `es`, `fr`, `sw`, `ar`, `ur`, `bg`, `ru`, `hi`) and uses five frozen orders and three seeds. All 15 runs have verified result ZIPs and executed notebooks, and the combined Kaggle analysis validates their frozen data-manifest hash. The pooled embedding-similarity correlation is positive (Spearman rho 0.4718, Holm-adjusted p 0.0455); URIEL+ syntactic distance is nominally associated (rho 0.3325, p 0.0318) but not after Holm correction (p 0.1908). These are pilot permutation results conditional on the observed forgetting matrix and do not establish robustness across training seeds. Arabic published-interference values use the pre-specified Phase 4 k=5 URIEL+ imputation and must be reported as imputed; Grambank is excluded for coverage. See the [combined analysis report and artifacts](./phase3/results/rq1_v2_combined_analysis_2026-11-24/README.md), [per-order run index](./phase3/results/rq1_v2_order_runs/README.md), [v2 input-preparation record](./phase3/results/rq1_v2_phase4_roster_input_prep_2026-11-20/README.md), [confirmatory result folder](./phase3/results/confirmatory_b0_seeds_43_44_2026-11-19/README.md), and [v2 workflow](./phase3/README.md#expanded-rq1-v2-workflow). |
| Phase 4: controlled ten-language prior comparison | **Initial pilot complete: 15/15; extension underway: 1/31 validated** | B2, B5, and M3 completed across all five frozen orders at seed 42; their results remain under the original protocol. In the approved extension, the single order-independent B1 run is validated at ACC 0.693; the other 30 extension records remain to be run. The full screen target is 46 records across two protocol IDs; preserve both hashes when analyzing. Current pilot ACC means across orders: B2 0.6807 (SD 0.0086), B5 0.6632 (SD 0.0137), M3 0.6858 (SD 0.0104). All results are descriptive single-seed evidence, not significance-backed evidence of seed robustness. See the [Phase 4 report and extension notebooks](./phase4/README.md) and [B1 execution record](./phase4/results/full_methods_seed42_b1_joint_2026-11-25/README.md). |

**Phase count:** The repository defines Phases 0–4 (five stages). Phase 0 is complete based on the reported paper/concept completion and archived English→Hindi experiment evidence. Phase 4's initial three-method pilot is complete, but the approved full-baseline single-seed screen remains in progress.

Phase 1 being accepted for project progression is **not** a claim that the original checkpoint pattern was reproduced.

## Phase 1 results

Both matched stream-order runs use 1,000 training and 500 test examples per
language, test-selection seed 42, training seeds 42–44, and
`meta-llama/Llama-3.2-1B`. The English → Hindi → German results are:

| Seed | ACC | BWT | FWT | Forgetting | Pattern passed |
|---:|---:|---:|---:|---:|:---:|
| 42 | 0.7093 | 0.1420 | 0.1880 | -0.0800 | No |
| 43 | 0.6787 | 0.1130 | 0.1780 | -0.1130 | No |
| 44 | 0.6853 | 0.1010 | 0.1730 | -0.0650 | No |
| Mean ACC | **0.6911** | — | — | — | **0/3 passed** |

Mean final accuracy by language: English 0.7440, Hindi 0.6507, German 0.6787.
See the [English → Hindi → German run summary](./phase1/results/xnli-en-hi-de-multiseed-train1000-test500-20261116T074727501956Z/README.md).

The completed Hindi → German → English group has:

| Seed | ACC | BWT | FWT | Forgetting | Pattern passed |
|---:|---:|---:|---:|---:|:---:|
| 42 | 0.7053 | 0.0640 | 0.2960 | -0.0140 | No |
| 43 | 0.7280 | 0.1180 | 0.2470 | -0.0380 | No |
| 44 | 0.6860 | 0.1230 | 0.1660 | -0.0440 | No |
| Mean ACC | **0.7064** | — | — | — | **0/3 passed** |

Use the [Phase 1 results index](./phase1/results/README.md) and each run's
summary as the result sources of truth. The mean ACC difference between the
matched orders is descriptive only; three seeds do not establish significance.
Keep the separate English → Hindi → German pilot and incomplete attempts
separate; their results are not interchangeable.

Both matched runs are complete. Their clean notebooks remain available as
templates: [EN → HI → DE Kaggle notebook](./phase1/notebooks/phase1_xnli_en_hi_de_multiseed_kaggle.ipynb)
and [HI → DE → EN Colab notebook](./phase1/notebooks/phase1_xnli_hi_de_en_sequential_wandb_colab.ipynb).
The existing 500/200 single-seed result is not a substitute for either matched
arm.

## Phase 2: baseline status

Current Phase 2 comparison target: five languages (`en`, `hi`, `es`, `de`, `sw`), 500 training and 200 test examples per language, with five fixed orders for order-sensitive methods. B1 is order-invariant and runs once. The configured B2 buffer is 125 examples (5% of the selected training set).

| Method | What exists now | Status / next action |
|---|---|---|
| **B0 — sequential fine-tuning, no replay** | Five completed orders, seed 42; mean ACC 0.5822 (SD 0.0484), BWT 0.0983 (SD 0.0447), FWT 0.1100 (SD 0.0274), F -0.0458 (SD 0.0475). | Complete for the current five-order protocol. All result records and prepared-data checksums validate locally. The public repo will contain the results and selection manifest, not upstream XNLI sentence text or model checkpoints. See the [B0 run record](./phase2/runs/b0_kaggle_2026-11-11/README.md). |
| **B1 — joint multilingual training** | One completed, current-protocol run: ACC 0.623; compact result and matching manifest are published. | Complete for B1's order-invariant single-run protocol. The adapter and fixed data remain local; do not mix with the older order-1 B1 record (ACC 0.671). |
| **B2 — Uniform Experience Replay** | Five completed orders, seed 42; mean ACC 0.6344 (SD 0.0143), BWT 0.0980 (SD 0.0275), FWT 0.1825 (SD 0.0299), F -0.0055 (SD 0.0172). | Complete for the current five-order protocol. All result records and prepared-data checksums validate locally. See the [B2 run record](./phase2/runs/b2_kaggle_2026-11-11/README.md). |
| **B3 — DER++** | Five completed orders, seed 42; mean ACC 0.4206 (SD 0.0079), BWT -0.0375 (SD 0.0328), FWT 0.0283 (SD 0.0145), F 0.0585 (SD 0.0260). | Complete for the current five-order protocol; compact results validate against the shared manifest. This is a documented candidate-label-logit adaptation, not full-vocabulary DER++. |
| **B4 — GCR-style adaptation** | Five orders; mean ACC 0.6464 (SD 0.0121), BWT 0.1183 (SD 0.0285), FWT 0.1815 (SD 0.0097), F -0.0250 (SD 0.0180). | Complete and validated. Reports the CountSketch/gradient-herding adaptation and added gradient-selection compute; not an exact paper reproduction. |
| **B5 — Embedding-Similarity Replay** | Five orders; mean ACC 0.6020 (SD 0.0271), BWT 0.0833 (SD 0.0342), FWT 0.1558 (SD 0.0144), F -0.0118 (SD 0.0466). | Complete and validated; frozen-base nearest-neighbor adaptation. |
| **B6 — LR-ADJUST** | Five orders; mean ACC 0.5610 (SD 0.0466), BWT 0.0943 (SD 0.0382), FWT 0.1085 (SD 0.0239), F -0.0313 (SD 0.0276). | Complete and validated; report the gamma and learning-rate floor as adaptation choices. |
| **B7 — COMPASS-inspired replay** | Five orders; mean ACC 0.6138 (SD 0.0090), BWT 0.0988 (SD 0.0106), FWT 0.1548 (SD 0.0089), F -0.0110 (SD 0.0187). | Complete and validated; semantic-cluster/target-distribution adaptation, not the paper's DoRA setup. |
| **B8 — Activation-anchor replay** | Five orders; mean ACC 0.6314 (SD 0.0154), BWT 0.0865 (SD 0.0127), FWT 0.1948 (SD 0.0163), F 0.0010 (SD 0.0116). | Complete and validated; distinguish the adaptation from the source paper's attention-head intervention and KL objective. |

For B2, the implemented policy samples uniformly from a stagewise reservoir containing only previously seen languages. Before each later stage it draws one replay example (with replacement) per current example; the current stage is added to the reservoir afterward. The first stage has no replay. The 125-example capacity and 1:1 replay ratio are protocol choices, and replay means B2 sees more total training-example exposures than B0.

The [Phase 2 run index](./phase2/runs/README.md) maps the canonical result records, manifests, summaries, and clean notebooks. Full Kaggle archives, fixed XNLI sentence data, and model checkpoints remain local. The local `phase2/artifacts/protocol_manifest.json` is a legacy manifest and must not be used to schedule or aggregate new runs. All nine selected result sets have current-protocol records and the same frozen data-manifest hash. These v1 runs are not compute-matched: replay adds example exposures and B4 adds gradient-selection work. The SDs are across five orders with one training seed per order; they are descriptive and not significance tests. Use a new protocol ID for any v2 rerun and do not mix it with v1.

## Key references

- [Research plan](./research_plan.md)
- [Phase 1 results index](./phase1/results/README.md)
- [Phase 2 README and run instructions](./phase2/README.md)
- [Phase 2 protocol configuration](./phase2/configs/pilot_protocol.json)
- [B0-B5 runner](./phase2/run_baseline.py)
- [B0 Kaggle notebook](./phase2/notebooks/phase2_b0_kaggle.ipynb)
- [B2 Kaggle notebook](./phase2/notebooks/phase2_b2_kaggle.ipynb)
- [Phase 2 run index](./phase2/runs/README.md)
- [B3 Kaggle notebook](./phase2/notebooks/phase2_b3_kaggle.ipynb)
- [B4 Kaggle notebook](./phase2/notebooks/phase2_b4_kaggle.ipynb)
- [B5 Kaggle notebook](./phase2/notebooks/phase2_b5_kaggle.ipynb)
- [Phase 3 README and inputs](./phase3/README.md)
- [Phase 3 RQ1 Kaggle notebook](./phase3/notebooks/phase3_rq1_prior_comparison.ipynb)
