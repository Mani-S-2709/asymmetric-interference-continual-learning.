# Phase 1: reusable multilingual NLI streams

Phase 1 uses a no-replay NLI run to validate the stream, training, evaluation,
metrics, and logging harness. A stream stage now declares its language tag,
task, Hugging Face dataset ID, optional dataset configuration, and optional
revision. Dataset IDs and language codes are not hard-coded in the generic
loader, so languages written in different scripts can share the same pipeline.

## English-Hindi-Bengali pilot

The prepared config uses:

| Stream position | Language | Hugging Face source | Config |
|---:|---|---|---|
| 1 | English (`en`) | `facebook/xnli` | `en` |
| 2 | Hindi (`hi`) | `AdaMLLab/indicxnli_repaired` | `hi` |
| 3 | Bengali (`bn`) | `AdaMLLab/indicxnli_repaired` | `bn` |

Both repositories are pinned to dataset commit revisions in
[`configs/en_hi_bn_indicxnli.json`](./configs/en_hi_bn_indicxnli.json).
Selected records are checked for the NLI fields `premise`, `hypothesis`, and
three-way labels; the runner uses the same entailment/neutral/contradiction
mapping for every source. It also checks that the fixed selected row indices
and label sequences agree across the three sources before training. It records
source revisions, fingerprints, exact selected row indices, and a
data-manifest SHA-256 alongside results.

The IndicXNLI dataset card describes Hindi and Bengali as translations of
English XNLI with matching split sizes. The repaired Hub copy is a community
conversion, not the original authors' upload. The original card's license
metadata and license text conflict, and the repaired copy does not declare a
clear license in its current Hub metadata. **Verify the permitted research and
publication use before running or distributing results.** Cite the IndicXNLI
paper when using that data.

## Run

On a CUDA machine with the Phase 1 dependencies installed and access to the
gated model:

```powershell
python phase1\scripts\run_xnli_en_hi_bn.py --preflight-only
python phase1\scripts\run_xnli_en_hi_bn.py
```

The first command loads the datasets and model and checks evaluation on a small
sample. The second runs seeds 42, 43, and 44. Set `HF_TOKEN` in the runtime
environment after accepting the model terms. Results and the selection
manifest are written under `phase1\results`; adapters are written under
`phase1\checkpoints`.

The generic entry point accepts another stream config:

```powershell
python phase1\scripts\run_multilingual_nli.py --config path\to\stream.json
```

Each stream entry may use a different dataset ID, config, and revision. Keep
the task's fields and label mapping compatible with the NLI formatter, or add
a task-specific formatter rather than silently coercing labels. The
`HuggingFaceStreamLoader` can also be constructed with an injected
`dataset_loader` for local or test datasets.

The legacy `XNLIStreamLoader` import remains as a compatibility subclass; new
streams should use `HuggingFaceStreamLoader`.

## Scientific interpretation

The checkpoint rule is not changed by this data-source update. A new
English-Hindi-Bengali run is a new experiment and must not be combined with
the earlier English-Hindi-German or Hindi-German-English results. A rerun
cannot guarantee the checkpoint will pass.
