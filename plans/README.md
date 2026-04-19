# Plans & experiment artifacts

This folder holds **archived experiment outputs** copied from the original CryptoDL workspace (not a formal “plan” format — there were no separate plan files in-repo).

- **`checkpoints_archive/`** — JSON result summaries from `checkpoints/*.json` (training metrics, test splits, hyperparameters).
- **`logs_archive/`** — Text logs from `logs_*.txt` and `nohup*.log` (training runs, feature ablations, etc.).

The main codebase lives at the repo root (`scripts/`, etc.).

**Security:** Never commit API keys or GitHub tokens. If any credential was ever pasted into a log, rotate it.
