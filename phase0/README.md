# Phase 0 — Research Onboarding & Continual Learning Foundations

## Project

**Cross-Lingual Continual Learning with Interference-Aware Replay**

Phase 0 is the research onboarding stage. The goal is to understand the continual-learning foundations, cross-lingual continual learning, cross-lingual interference, replay methods, and language-similarity priors before implementing the main experimental framework.

---

# 1. Phase 0 Goals

By the end of Phase 0, I should be able to:

* Explain continual learning.
* Explain catastrophic forgetting.
* Explain forward transfer and backward transfer.
* Explain cross-lingual continual learning.
* Explain why language order can affect continual-learning performance.
* Explain experience replay.
* Explain why replay sample selection matters.
* Explain cross-lingual interference.
* Explain why interference can be asymmetric between languages.
* Explain the motivation for using language similarity/typology as a prior.
* Reproduce a small continual-learning experiment showing forgetting.
* Explain how the Phase 0 concepts connect to the later research phases.

---

# 2. Environment Checklist

* [x] VS Code project opened
* [x] Python virtual environment created
* [x] Python version verified
* [x] NVIDIA GPU detected
* [x] CUDA-enabled PyTorch installed
* [x] PyTorch CUDA availability verified
* [x] GPU computation tested
* [x] Phase 0 directory created
* [x] Reading-notes directory created
* [x] Experiments directory created
* [x] Results directory created

Current hardware:

* GPU: NVIDIA GeForce RTX 2050
* VRAM: 4 GB
* PyTorch: 2.14.1+cu130
* CUDA runtime: 13.0

Verification note (2026-11-25): the repository's current `.venv` could not
import PyTorch, so the toy experiment could not be rerun in this environment.
The hardware checklist above records the earlier environment setup.

---

# 3. Phase 0 Directory Structure

```text
phase0/
│
├── README.md
│
├── reading_notes/
│
├── experiments/
│
└── results/
```

---

# 4. Mandatory Papers

## Paper 1 — Cross-Lingual Continual Learning

**M'hamdi, Ren & May (2023)**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Cross-lingual continual learning problem
* Continual-learning setup
* Catastrophic forgetting
* Forward/backward transfer
* CFT
* CBT
* MFT
* MBT
* Experimental protocol

Notes:

`reading_notes/01_mhamdi_2023.md`

---

## Paper 2 — Massively Multilingual Continual Learning

**Winata et al. (2023)**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Massively multilingual continual learning
* Language streams
* Catastrophic forgetting
* LR-ADJUST
* Experimental design
* 51-language setting

Notes:

`reading_notes/02_winata_2023.md`

---

## Paper 3 — Cross-Lingual Interference

**Alastruey et al. (2025)**

**Interference Matrix: Quantifying Cross-Lingual Interference in Transformer Encoders**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Interference matrix
* Directional interference
* Asymmetric interference
* Language pairs
* Script effects
* Relationship between language similarity and interference
* Connection to the research hypothesis

Notes:

`reading_notes/03_alastruey_2025.md`

---

## Paper 4 — COMPASS

**Flynn (2026)**

**COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Multilingual PEFT
* Semantic sampling
* Replay/sample selection
* Continual multilingual learning
* Comparison with the proposed research direction

Notes:

`reading_notes/04_compass_2026.md`

---

## Paper 5 — GCR

**Tiwari et al. (2022)**

**GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Gradient-based replay selection
* Coresets
* Replay buffer construction
* Gradient similarity/diversity
* Connection to the GCR baseline

Notes:

`reading_notes/05_gcr_2022.md`

---

## Paper 6 — DER++

**Buzzega et al. (2020)**

**Dark Experience for General Continual Learning**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Experience replay
* Logit replay
* Dark knowledge
* DER
* DER++
* Replay buffer
* Connection to the DER++ baseline

Notes:

`reading_notes/06_derpp_2020.md`

---

## Paper 7 — EWC

**Kirkpatrick et al. (2017)**

**Overcoming Catastrophic Forgetting in Neural Networks**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Catastrophic forgetting
* Stability vs plasticity
* Parameter importance
* Fisher information
* Elastic Weight Consolidation
* Classical continual learning

Notes:

`reading_notes/07_ewc_2017.md`

---

## Paper 8 — URIEL / lang2vec

**Littell et al. (2017)**

Status: [x] Completed per researcher report; notes not recorded

Focus:

* Language representations
* Typological features
* Geographic features
* Genetic features
* Language vectors
* lang2vec
* Using language similarity as a prior

Notes:

`reading_notes/08_uriel_2017.md`

---

# 5. Paper Reading Template

For every paper, answer:

## 1. Research Problem

What problem is the paper trying to solve?

## 2. Motivation

Why is this problem important?

## 3. Main Idea

What is the central idea of the paper?

## 4. Method

How does the proposed method work?

## 5. Dataset / Languages

What datasets, languages, tasks, or models were used?

## 6. Experimental Setup

How were the experiments conducted?

## 7. Metrics

What metrics were used?

## 8. Main Results

What did the experiments demonstrate?

## 9. Important Equations

Record equations that are necessary for understanding the method.

## 10. Limitations

What limitations or open questions remain?

## 11. What I Learned

Write the most important lessons in your own words.

## 12. Connection to My Project

How does this paper affect my research design?

## 13. Questions

What do I still not understand?

---

# 6. Reading Order

The recommended order is:

```text
01 M'hamdi et al. (2023)
        ↓
02 Winata et al. (2023)
        ↓
03 Alastruey et al. (2025)
        ↓
04 COMPASS
        ↓
05 GCR
        ↓
06 DER++
        ↓
07 EWC
        ↓
08 URIEL / lang2vec
```

The first three papers are the highest-priority papers because they establish the research problem and motivate the interference-aware direction.

---

# 7. Toy Continual-Learning Experiment

After the paper-reading foundation, implement a small continual-learning experiment.

The experiment should contain three sequential tasks:

```text
Task A
  ↓
Train
  ↓
Evaluate A
  ↓
Task B
  ↓
Train
  ↓
Evaluate A + B
  ↓
Task C
  ↓
Train
  ↓
Evaluate A + B + C
```

The purpose is to observe catastrophic forgetting directly.

The experiment should record:

* Accuracy on Task A after Task A
* Accuracy on Task A after Task B
* Accuracy on Task A after Task C
* Accuracy on Task B after Task B
* Accuracy on Task B after Task C
* Accuracy on Task C after Task C

Save experiment code in:

```text
experiments/
```

Save results in:

```text
results/
```

---

# 8. Phase 0 Metrics

At minimum, understand:

### Average Accuracy

Performance across the tasks/languages after continual learning.

### Backward Transfer

How learning later tasks affects previously learned tasks.

### Forward Transfer

How knowledge from earlier tasks affects learning of future tasks.

### Forgetting

Performance degradation on previously learned tasks after subsequent learning.

For the later cross-lingual experiments, also understand:

* CFT
* CBT
* MFT
* MBT

---

# 9. Phase 0 Completion Gate

Phase 0 is complete only when:

## Research Understanding

These items are marked complete per the researcher's self-report; this
checklist is not an independent assessment.

* [x] I can define continual learning.
* [x] I can explain catastrophic forgetting.
* [x] I understand forward transfer.
* [x] I understand backward transfer.
* [x] I understand experience replay.
* [x] I understand replay-buffer selection.
* [x] I understand cross-lingual continual learning.
* [x] I understand cross-lingual interference.
* [x] I understand why interference may be asymmetric.
* [x] I understand the motivation for a language-similarity prior.

## Paper Reading

* [x] Paper 1 completed (researcher-reported; notes not recorded)
* [x] Paper 2 completed (researcher-reported; notes not recorded)
* [x] Paper 3 completed (researcher-reported; notes not recorded)
* [x] Paper 4 completed (researcher-reported; notes not recorded)
* [x] Paper 5 completed (researcher-reported; notes not recorded)
* [x] Paper 6 completed (researcher-reported; notes not recorded)
* [x] Paper 7 completed (researcher-reported; notes not recorded)
* [x] Paper 8 completed (researcher-reported; notes not recorded)

## Engineering

* [x] Toy continual-learning experiment implemented (`experiments/tiny_cl.py`)
* [x] Sequential training is implemented; researcher reports execution
* [x] Evaluation after every task is implemented
* [x] Earlier-task accuracy is evaluated after later tasks
* [ ] Results are saved as a project artifact
* [ ] Experiment rerun is verified in the current project environment

The checked-in toy script trains on three sequential tasks and prints accuracy
for every task seen after each stage. The researcher reports that the experiment
was executed, but no `phase0/results` directory or saved output is present in
the project. A rerun attempted on 2026-11-25 stopped with
`ModuleNotFoundError: No module named 'torch'` in the repository `.venv`.
Save the executed metrics and make PyTorch available in the active environment
before marking the final two engineering items complete.

## Research Connection

These items are also marked complete per the researcher's self-report.

* [x] I can explain why replay may reduce forgetting.
* [x] I can explain why replay selection matters.
* [x] I can explain why language relationships may help replay selection.
* [x] I can explain how the Phase 0 findings motivate Phase 1.

---

# 10. Phase 0 sequencing note

Phase 0 was originally intended to precede the full research experiments.
Phases 1–4 have since been executed and are documented in
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md). The paper-reading and concept
checklists are marked complete per the researcher's report; the reading-note
files remain unfilled. Keep Phase 0 in progress until the toy run's results are
saved and its rerun is verified.

---

# 11. Current Status

**Phase 0 status:** In progress — all eight paper readings and the concepts
checklist are marked complete per researcher report. The toy experiment source
is present and execution is reported, but saved results are absent and a rerun
is blocked by missing PyTorch in the current `.venv`.

**Overall project status:** Phase 1–3 experimental records and the initial
Phase 4 pilot are complete; an approved Phase 4 baseline extension remains in
progress. Phase 0 also remains in progress pending saved toy results and a
verified rerun. See
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md) for results, limitations, and
pending decisions.

**Next Phase 0 step:** Save the toy experiment's run metrics, restore PyTorch
in the active environment, and verify that the experiment can be rerun.
