# Phase 4 groundwork

`interference_replay.py` provides an isolated M3 replay selector for signed,
directed interference scores. Matrix rows are past languages and columns are
incoming languages; negative values become replay risk via `max(0, -score)`.
When every eligible risk is zero, language weights fall back to uniform.

This component does not choose a Phase 3 prior or freeze the Phase 4 stream.
The controlled language list, dataset compatibility, and final protocol remain
to be resolved before building or running complete Phase 4 experiments.

Run its focused tests from the repository root with:

```powershell
python -m pytest phase4\tests\test_interference_replay.py
```
