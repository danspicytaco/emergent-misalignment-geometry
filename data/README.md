# Research data

The retained public evidence is hosted at [danspicytaco/emergent-misalignment-geometry](https://huggingface.co/datasets/danspicytaco/emergent-misalignment-geometry). **All 44,363 included files (25,040,396,020 bytes) passed complete independent size/SHA256 read-back**, and an isolated candidate fetched the analysis profile and reproduced all seven numerical anchors without access to the original checkout. See [verification metadata](verification.json).

```bash
.venv/bin/python scripts/data/fetch.py --profile legacy-analysis
```

`sources.json` pins the full immutable revision, inventory hash and exact profile counts. The full inventory is hosted on HF and cached as ignored `.cache/legacy-migration.json`; it is not part of the Git code tree.

- Download cache: `artifacts/hf/` (override `--cache-root` / `EM_HF_CACHE`).
- Analysis inputs: `artifacts/hf/legacy/2026-11-16/tree/data/` (override explicit input arguments / `EM_DATA_DIR`).
- New outputs: `artifacts/runs/` or `EM_OUTPUT_DIR`.

The verified local raw-data cache and all 37 unmirrored/withheld files were subsequently deleted at the owner's explicit request. Fetch included evidence from HF when needed; omitted source training inputs must come from upstream or historical Git copies where present. Untracked operational logs/backups are not recoverable from this release. Git history remains unchanged. Successful verification concerns the included inventory, not withheld files or scientific validity.

Use `legacy-activations --model 14b` only when that tensor partition is needed; the analysis profile does not fetch every response/tensor. The training profile is empty because raw training inputs are upstream/local references. See [provenance](../docs/data-provenance.md) and [storage](../docs/data-storage.md).
