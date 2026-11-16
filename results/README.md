# Retained results

Canonical small outputs from the historical reproduction and robustness studies. These are exploratory results, not confirmation of broad misalignment, scaling or monitoring effectiveness.

| Folder | Contents |
|---|---|
| [legacy/binning](legacy/binning/) | Bin-size/layer sensitivity tables and plots |
| [legacy/scaling](legacy/scaling/) | Original medical size-ladder figures |
| [legacy/figures](legacy/figures/) | Base-model, distance, lexical and layer-shape figures |
| [legacy/stress_tests](legacy/stress_tests/) | Coherence/selection tables and figures |
| [legacy/stress_tests_judges](legacy/stress_tests_judges/) | Judge-scheme and topic-control summaries/figures |

Files under `v1_flawed_binned_ci` preserve superseded confidence intervals and must not be quoted as validated uncertainty. Raw judgments, per-prompt labels/text and frozen inputs are retained in the HF snapshot/local cache, not duplicated in the code tree.

See [current evidence](../docs/results.md), [study notes](../docs/studies/README.md), and the [historical full write-up](../docs/archive/stress-tests-writeup-2026-10.md). Replay into a new `artifacts/` output directory; do not overwrite these retained results.
