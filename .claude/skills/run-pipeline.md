---
description: Run one or more pipeline stages for the process control project
---

Run the requested pipeline stage(s) using make targets.

Available stages in order:
1. `make simulate` — generate process data
2. `make validate` — validate data schema
3. `make graph` — discover causal DAG
4. `make train` — train causal model + detector
5. `make evaluate` — compute SHD and refutation metrics
6. `make serve` — launch Dash dashboard
7. `make all` — run stages 1–5 end-to-end

If the user asks to run the full pipeline, use `make all`.
If the user asks to retrain, use `make train`.
If the user asks to start the dashboard or server, use `make serve`.

Always run from the project root directory.
Show the make output so the user can see progress logs.
