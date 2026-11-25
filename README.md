# Geometry of Emergent Misalignment

Research on model-size dependence of geometry-based prediction and the safety–utility of centroid monitoring. This began as an [ML4Good](https://www.ml4good.org/) reproduction of Li, Dai, Wang and Tan's [*Emergent Misalignment Is Not Magical*](https://arxiv.org/abs/2608.29118). Upstream-derived code and attribution remain in `em-not-magical/` and Git history.

## Scope

- **B0:** qualify coherent, reproducible cross-domain harmful behaviour against the instruction-model baseline.
- **R0:** reproduce the relevant centroid-distance association.
- **S1:** test whether predictive association strengthens with model size.
- **M1:** evaluate pre-generation monitoring at matched legitimate-use costs.

Existing checkpoints have not passed B0. Exploratory correlations are not a universal scaling law or a proven safeguard. New experiment methodology remains open; no model is downloaded or run by the setup below.

## CPU setup and retained-data replay

Bulk evidence is hosted on HF rather than in Git history. Use a normal checkout, then the tested Python 3.13 environment:

```bash
git clone https://github.com/danspicytaco/emergent-misalignment-geometry.git
cd emergent-misalignment-geometry
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements/analysis.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/data/fetch.py --profile legacy-analysis
.venv/bin/python scripts/reproduction/replay_anchors.py --data-root artifacts/hf/legacy/2026-11-16/tree/data --out-dir artifacts/replay --anchor all
```

The public [HF snapshot](https://huggingface.co/datasets/danspicytaco/emergent-misalignment-geometry) is pinned in `data/sources.json`. All 44,363 included files were independently downloaded and matched by size/SHA256; see [verification metadata](data/verification.json). The full inventory is fetched explicitly and cached, not tracked in Git. Source inputs withheld for rights/privacy remain local/upstream references.

| Path | Purpose |
|---|---|
| `src/em_geometry/` | Shared CPU paths, records, metrics and artifact helpers |
| `scripts/data/` | Explicit inventory, publication, retrieval, verification and catalogue commands |
| `scripts/reproduction/` | Reproduction and saved-distance replay |
| `scripts/analysis/` | Figures, binning, judge and stress-test analyses |
| `configs/`, `requirements/`, `tests/` | Scientific settings, environments and offline checks |
| `results/legacy/` | Canonical small tables, figure data and figures |
| `docs/` | Current goals, methodology, runbook, studies and clearly marked archives |
| `data/` | Small source/verification metadata; ignored inventory cache |
| `artifacts/`, `.scratch/` | Ignored retained evidence, run outputs and disposable work |

Only working consumers are introduced; no qualification/scaling/monitoring stubs imply completed experiments. Bulk data was removed from Git history; commit IDs changed, while code history and attribution remain. Existing collaborators should re-clone rather than merge pre-rewrite history. Local raw-data and refactor scratch copies were deleted after verification; fetch only the inputs needed for an analysis.

Start with [goals](docs/goals-and-hypotheses.md), [methodology](docs/methodology.md), [results](docs/results.md), [storage](docs/data-storage.md) and the [runbook](docs/runbook.md). Archived operational commands are not execution authority.

**Content warning:** research artifacts contain harmful, offensive and medically unsafe model outputs. They are not advice or a deployment-ready safeguard. Upstream material is not relicensed by this project.
