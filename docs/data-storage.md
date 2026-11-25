# Storage and verification

- **GitHub:** source, configs, tests, documentation and small canonical results.
- **Public HF dataset:** included retained responses, scores, centroids and activation tensors.
- **Ignored artifacts:** local materialisation, withheld originals and new outputs.

`data/sources.json` pins the public dataset at an immutable revision. The initial snapshot retains original file bytes/relative paths under `legacy/2026-11-16/tree/`. Complete independent content read-back and isolated numerical replay passed; the small [verification record](../data/verification.json) states their exact scope.

Following complete verification, the owner explicitly requested deletion of local research-data copies, including all 37 unmirrored/withheld files, duplicate figures and operational scratch. Included evidence remains in the pinned HF snapshot; the full read-back report is linked from the verification record. Explicit fetches recreate `artifacts/hf/legacy/2026-11-16/tree/` as needed. Bulk data was subsequently removed from Git history, while source, attribution and small-result history were retained. Collaborators should re-clone rather than merge pre-rewrite history.

The complete inventory is hosted on HF and cached as ignored `data/.cache/legacy-migration.json`. Explicit fetches validate it before selecting files; imports never download automatically. A successful analysis-profile fetch is not itself full transfer verification—the separate all-file read-back was performed.

Upstream terms still apply, and a gated mirror does not confer redistribution rights. No private credentials, billing metadata or unreviewed operational archives are published as research evidence. See [provenance](data-provenance.md). Mutable buckets are not the canonical evidence source.

Code is published under `danspicytaco/emergent-misalignment-geometry`, preserving the original repository identity, code history and attribution; bulk-data removal rewrote commit IDs. The local repo directory and Hermes project are named `emergent-misalignment-geometry`. No new model experiments or paid compute were run.
