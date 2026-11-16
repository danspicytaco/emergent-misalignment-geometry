# Data provenance

## Sources

| Evidence | Source and boundary |
|---|---|
| Original reproduction code | [ChicagoHAI/em-not-magical](https://github.com/ChicagoHAI/em-not-magical), vendored from `28f0782` with local patches; not newly authored code |
| Reconstructed Arena prompts | `agie-ai/lmsys-chatbot_arena_conversations`, a mirror of the canonical [LMSYS dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations); original terms and privacy restrictions still apply |
| Medical training input | [geodesic-research/sfm-emergent-misalignment-training-data](https://huggingface.co/datasets/geodesic-research/sfm-emergent-misalignment-training-data), medical subset; explicit redistribution licence not identified during this migration |
| Released evaluation prompts | Authors' `sec3/chatbot_verified_prompts.json`, retained verbatim in the released-14B evidence |
| Generated responses, activations and judgments | This project's runs, with checkpoint/request/execution differences recorded in the retained notes |

The reconstruction retained 638 prompts; the later released set contains 822, with 38 exact overlaps. This is a dataset distinction, not proof that execution matches the paper. The newer released-prompt run used a different GPU stack and existing matched centroid; its baseline outputs have not all been judged. The GPT bridge and repeated judging hold saved responses fixed.

Rehosting is not relicensing. Training/source data without clear redistribution rights is referenced upstream, not automatically published under this project's name. Local withheld copies were deleted during explicitly authorised cleanup; historical tracked copies may remain in Git history. The LMSYS source identifies a Creative Commons licence only generically and prohibits attempts to identify individuals; do not treat its mirror as bypassing those terms. Operational logs, credential material and private metadata are excluded from research publication.

The exact HF snapshot revision, included paths and omissions belong in `data/sources.json` and the migration inventory. See the [historical source audit](archive/data-provenance-2026-10.md) for the original reconstruction decisions; its claims that the authors' prompts remain unavailable are superseded.
