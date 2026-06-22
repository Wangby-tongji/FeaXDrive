# Train and Evaluation

This file is kept for compatibility with the original ReCogDrive documentation layout.

For the current FeaXDrive codebase, use:

- `docs/FEAXDRIVE_REPRODUCTION.md` for cache, training, and evaluation commands.
- `docs/FEAXDRIVE_MODEL_CODE_GUIDE.md` for model architecture and code-chain analysis.
- `docs/FEAXDRIVE_FAGRPO_CHAIN.md` for the isolated FA-GRPO training reward path.

The older ReCogDrive scripts in this repository are preserved only as reference/compatibility code. The verified FeaXDrive reproduction entrypoints are:

```text
scripts/repro/slurm/submit_cache_feaxdrive.slurm
scripts/repro/slurm/submit_train_feaxdrive.slurm
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```
