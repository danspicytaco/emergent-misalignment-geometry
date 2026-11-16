# Current evidence

All findings remain exploratory until scientific and human validation. The refactor preserves calculations; it does not promote their claims.

- A negative prompt-level distance–misalignment association is present in the existing medical fine-tunes and survives the released-prompt 14B rerun.
- Fixed-middle-layer last-token correlations favour the 32B versus 3B endpoint, but are not monotonic across the ladder. Response-mean extraction does not show the same size pattern.
- Binned selected-layer correlations and unbinned fixed-layer correlations are different estimands. The older paired panels also use different statistics (mean Spearman/Pearson versus Spearman alone), so the visual gap is not exclusively binning and selection.
- Released prompts, GPT-5.4 rejudging and three fresh judge passes are retained in the pinned HF snapshot. Judge variation changes sweep optima; it does not establish the cause of every discrepancy with the paper.
- The existing checkpoints have not passed the new broad-misalignment qualification criterion. No effective monitor or universal scaling law has been demonstrated.

Use the saved-data replay command to recompute the numerical anchors from retained CSV/array/raw-score inputs. The [historical chronology](archive/results-2026-10.md), [binning study](studies/binning.md) and [judge studies](studies/judges/jev_outcome_judge.md) retain the original evidence and caveats. Some original paths are historical; retained included data now belongs in the pinned HF snapshot and can be fetched explicitly after local cleanup.
