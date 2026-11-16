# Historical environments

These are observed execution stacks, not one compatible requirements set:

- Camp local analysis: torch 2.10.0, transformers 5.8.0, with no local vLLM.
- Camp GPU generation/extraction: vLLM 0.19.0, torch 2.10.0+cu128, transformers 4.57.6; the original transformers 5.8 pin conflicted.
- Released 14B rerun: vLLM 0.11.0, transformers 4.57.1, torch 2.8.0, BF16/greedy; see retained `data/released_14b/notes.md`.

The new CPU package is tested separately. Do not silently reproduce old results with a different GPU environment or install these mutually conflicting versions together. A new training environment will be specified only after the training recipe is selected.
