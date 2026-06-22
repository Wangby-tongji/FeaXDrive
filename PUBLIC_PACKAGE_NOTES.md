# FeaXDrive GitHub-ready public package

Created from user-provided package: `FeaXDrive_github_ready_20260622_161608.tar.gz`.

This variant removes pack-time metadata files and duplicate root README files, replaces local absolute paths with placeholders, and keeps the verified FeaXDrive cache/train/eval source chains.

## Verified entrypoints

- `scripts/repro/slurm/submit_cache_feaxdrive.slurm`
- `scripts/repro/slurm/submit_train_feaxdrive.slurm`
- `scripts/repro/slurm/submit_eval_feaxdrive.slurm`

## Notes

- Runtime outputs, caches, checkpoints, and datasets are intentionally excluded.
- The user should edit placeholder paths such as `/path/to/FeaXDrive`, `/path/to/navdata`, and `/path/to/ReCogDrive-VLM-2B` before running.
- `docs/Train_Eval.md` is kept as a compatibility pointer to the current FeaXDrive documentation.
