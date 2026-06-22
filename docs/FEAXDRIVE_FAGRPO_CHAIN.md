# FeaXDrive FA-GRPO Chain

This document records the isolated FA-GRPO training chain used in the current FeaXDrive reproduction.

---

## 1. Stable chains left unchanged

The following chains should not be modified by FA-GRPO reward experiments.

### IL / dyn training

```text
submit_train_feaxdrive.slurm
-> MODE=eps / dt / dyn
-> agent=feaxdrive_v4drive_agent
-> planner_variant=standard
-> feaxdrive_diffusion_planner.py
```

### Final evaluation

```text
submit_eval_feaxdrive.slurm
-> run_pdm_score_feaxdrive.py
-> pdm_score.py
-> pdm_scorer.py
-> pdm_comfort_metrics.py
```

Final evaluation may use data-map guidance for `drivedyn` or `fa`, but scoring remains standard.

---

## 2. FA-GRPO training chain

```text
submit_train_feaxdrive.slurm MODE=fa
-> agent=feaxdrive_v4drive_fagrpo_agent
-> feaxdrive_agent_entry_v4drive_fagrpo.py
-> base FeaXDrive agent
-> planner_variant=fagrpo
-> feaxdrive_diffusion_planner_fagrpo.py
-> pdm_score_feaxdrive_fagrpo.py
-> feaxdrive_fagrpo_scorer.py
-> feaxdrive_fagrpo_comfort_metrics.py
```

---

## 3. Default reward weights

```text
progress_weight = 10.0
ttc_weight = 5.0
comfortable_weight = 10.0
gamma_denoising = 0.6
bc_coeff = 0.1
```

Slurm override variables:

```bash
FA_REWARD_PROGRESS_WEIGHT=...
FA_REWARD_TTC_WEIGHT=...
FA_REWARD_COMFORTABLE_WEIGHT=...
FA_GAMMA_DENOISING=...
FA_BC_COEFF=...
```

---

## 4. Required inputs

```text
reference policy checkpoint = dyn checkpoint
hidden cache = split-aligned hidden-state cache
metric cache = split-aligned metric cache
```

For the current Impromptu FA-GRPO reproduction:

```bash
CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt

IMP_HIDDEN_CACHE=/path/to/data/navcache/imp_hidden_cache_orin

IMP_METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache
```

---

## 5. Train/eval distinction

FA-GRPO training uses the isolated FA-GRPO scorer.

FA-GRPO final evaluation uses:

```text
standard pdm_scorer.py
standard pdm_comfort_metrics.py
```

Do not replace standard final evaluation files with FA-GRPO variants.
