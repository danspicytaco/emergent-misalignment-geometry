# Runbook: saved-data analysis

These commands retrieve retained evidence and recompute statistics. They do not generate model responses or buy compute.

```bash
.venv/bin/python scripts/data/fetch.py --profile legacy-analysis
.venv/bin/python scripts/reproduction/replay_anchors.py --data-root artifacts/hf/legacy/2026-11-16/tree/data --out-dir artifacts/replay --anchor all
```

All seven anchors must pass; missing files are errors, not skipped checks. Full-precision GPT sweeps retain the documented float32/default-sort/full-bin conventions, separately from rounded CSV protocols. Passing saved-distance replay does not reproduce acquisition, extraction or prove scientific hypotheses.

For plotting saved judge studies, pass separate input and output directories (`--inputs`, `--out`). `plot_stress_tests.py --results` names saved statistics, while `--out` names new figures. Never write into the retained input cache or `results/legacy/`.

New helpers default to the pinned HF cache. Existing shell/dotenv `EM_DATA_DIR` values may still name the former root `data/` folder; unset them for the default or explicitly point them to the retained data directory. Credential files were not modified.

Generation, centroid extraction and legacy API scripts require separately approved models, environments and spend ceilings. Upstream training inputs withheld from HF must be obtained from their original source or a historical Git copy where present; the earlier local-only copies were explicitly deleted. Historical [archive](archive/README.md) commands are not standing orders; centroid file arguments are literal paths, not automatically relative to `EM_DATA_DIR`.
