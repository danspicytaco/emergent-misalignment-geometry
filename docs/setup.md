# Setup

The CPU environment and numerical replay were exercised with Python 3.13.14 on macOS:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements/analysis.txt
.venv/bin/python -m pytest
```

Run installation from this checkout: the editable dependency in the requirements file is relative to the shell working directory. Analysis/HF client dependencies are separate from torch, transformers, vLLM and model weights. Data retrieval is explicit; see the [runbook](runbook.md).

Historical GPU stacks are documented in [historical environments](../requirements/historical-environments.md); they are not one compatible requirements set. New training methods, models, providers and paid calls still require methodology/budget approval.

Shared imports do not load credentials, initialise models or contact a provider. Upstream-derived acquisition scripts may load dotenv; they are legacy GPU/paid entry points, not part of the CPU quickstart. Credentials remain outside Git and exported evidence.
