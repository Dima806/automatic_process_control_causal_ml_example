---
description: Run all code quality checks for the project
---

Run the full quality gate in order:

1. `make lint` — ruff check --fix, ruff format, ty type check
2. `make test` — pytest with coverage report
3. `make complexity` — radon cyclomatic complexity (B-grade and below)
4. `make security` — bandit medium/high severity scan
5. `make audit` — pip-audit vulnerability scan

Report any failures clearly, grouping by check type.
Fix lint and type errors automatically if asked.
For test failures, show the full traceback and suggest a fix.
For security findings, explain the risk and suggest a code-level fix (not a nosec suppression).
