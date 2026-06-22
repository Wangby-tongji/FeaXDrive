# FeaXDrive Model Code Guide

This document explains the current FeaXDrive model code and verified cache / train / evaluation chains. It is intended for future development.

---

## 1. Architecture overview

FeaXDrive follows the ReCogDrive idea of injecting VLM driving priors into a continuous diffusion planner, but the current cleaned branch focuses on a NAVSIM reproduction and development chain:

```text
front camera image
-> InternVL / ReCogDrive VLM
-> hidden state cache
-> DiT diffusion planner
-> trajectory
-> PDM evaluation
```

The current core additions are:

1. x0-pred diffusion planner.
2. dynamics feasibility training (`dyn`).
3. data-map drivable guidance at inference (`drivedyn`).
4. isolated FA-GRPO reward training chain.
5. standard final PDM evaluation preserved unchanged.

---

## 2. Core files

### 2.1 Agent

```text
navsim/agents/feaxdrive/feaxdrive_agent_entry_v4drive.py
```

Main responsibilities:

- Instantiate feature and target builders.
- Choose planner class according to `planner_variant`.
- Load checkpoint.
- Provide `forward()` for training.
- Provide `compute_trajectory()` for evaluation.
- Inject drivable SDF context with `set_drivable_sdf_context()`.

Planner routing:

```text
planner_variant=standard
  -> feaxdrive_diffusion_planner.py

planner_variant=fagrpo
  -> feaxdrive_diffusion_planner_fagrpo.py
```

### 2.2 Feature builder

```text
navsim/agents/feaxdrive/feaxdrive_features.py
```

Main responsibilities:

- Read history trajectory.
- Encode command and ego status.
- Generate or read InternVL hidden state.
- Produce `internvl_feature.gz` and `trajectory_target.gz`.

Modes:

| Mode | Configuration | Behavior |
|---|---|---|
| hidden cache generation | `cache_hidden_state=True`, `cache_mode=True` | initialize VLM and write `last_hidden_state` |
| cached training | `cache_hidden_state=True`, `cache_mode=False` | read cached `last_hidden_state` |
| online inference | `cache_hidden_state=False` | process image online if required |

### 2.3 VLM backbone

```text
navsim/agents/feaxdrive/feaxdrive_backbone.py
```

Responsibilities:

- Load InternVL / ReCogDrive VLM checkpoint.
- Preprocess image patches.
- Run prompt-conditioned forward.
- Return hidden states for planner conditioning.

### 2.4 Diffusion planner

```text
navsim/agents/feaxdrive/feaxdrive_diffusion_planner.py
```

Responsibilities:

- DiT diffusion action generation.
- Support `eps` and `x0` prediction.
- DDIM sampling.
- Standard supervised diffusion loss.
- Dynamics feasibility regularization.
- Data-map drivable guidance during evaluation.

Important concepts:

```text
eps     = epsilon prediction
dt/x0   = direct x0 trajectory prediction
dyn     = x0 + dynamics feasibility
drivedyn= dyn checkpoint + drivable guidance at eval
```

### 2.5 FA-GRPO planner

```text
navsim/agents/feaxdrive/feaxdrive_diffusion_planner_fagrpo.py
```

Responsibilities:

- Load reference policy.
- Sample candidate trajectories.
- Compute FA-GRPO reward.
- Compute GRPO policy loss and BC regularization.
- Use isolated FA-GRPO PDM scorer.

Default reward:

```text
progress_weight=10.0
ttc_weight=5.0
comfortable_weight=10.0
gamma_denoising=0.6
bc_coeff=0.1
```

### 2.6 Projector

```text
navsim/agents/feaxdrive/trajectory_projector.py
```

Responsibilities:

- Dynamics projection and violation statistics.
- Drivable SDF projection.
- Footprint-aware drivable-area guidance.
- Metric-cache/map-based SDF context construction.

Do not reintroduce `use_constraint_projection`. The current official chain uses dynamics feasibility and data-map guidance instead.

---

## 3. Cache chain

### 3.1 Correct hidden-state cache chain

```text
submit_cache_feaxdrive.slurm
-> run_dataset_caching_multi_node.py
-> SceneLoader(..., load_image_path=True)
-> Dataset.cache_dataset()
-> FeaXDriveFeatureBuilder.compute_features()
-> InternVL hidden state
-> internvl_feature.gz
```

Why this matters:

- `FeaXDriveFeatureBuilder` expects image paths when calling InternVL image loading.
- `load_image_path=True` keeps image fields as paths.
- If training entry is used for caching, image fields may become NumPy arrays and trigger path errors.

### 3.2 Training cache read chain

```text
submit_train_feaxdrive.slurm
-> run_training_feaxdrive_rl.py
-> CacheOnlyDataset
-> internvl_feature.gz + trajectory_target.gz
-> AgentLightningDiT
```

Recommended training setting:

```text
use_cache_without_dataset=true
```

This avoids online VLM computation.

### 3.3 Metric cache chain

```text
submit_cache_feaxdrive.slurm
-> run_metric_caching.py
-> MetricCacheProcessor
-> metric_cache.pkl
```

Metric cache is used by:

- FA-GRPO reward function during training.
- Evaluation PDM scoring and drivable SDF context.

---

## 4. Training chain

### 4.1 Supervised IL / dyn

```text
run_training_feaxdrive_rl.py
-> instantiate agent
-> CacheOnlyDataset
-> AgentLightningDiT
-> FeaXDriveAgent.forward()
-> FeaXDriveDiffusionPlanner
-> supervised diffusion loss
```

`dyn` adds dynamics feasibility regularization.

### 4.2 FA-GRPO

```text
run_training_feaxdrive_rl.py
-> feaxdrive_v4drive_fagrpo_agent
-> planner_variant=fagrpo
-> FeaXDriveFAGRPODiffusionPlanner
-> reference policy
-> metric cache reward
-> GRPO loss + BC loss
```

Reward scorer path:

```text
pdm_score_feaxdrive_fagrpo.py
-> feaxdrive_fagrpo_scorer.py
-> feaxdrive_fagrpo_comfort_metrics.py
```

---

## 5. Evaluation chain

```text
submit_eval_feaxdrive.slurm
-> run_pdm_score_feaxdrive.py
-> MetricCacheLoader
-> SceneLoader
-> agent.compute_trajectory()
-> standard pdm_score()
-> standard PDMScorer
```

Final scorer path:

```text
pdm_score.py
-> pdm_scorer.py
-> pdm_comfort_metrics.py
```

FA-GRPO final evaluation does not use FA-GRPO scorer. It loads an FA-GRPO-trained checkpoint, but scoring remains standard.

---

## 6. Drivable guidance implementation

### 6.1 When enabled

`drivedyn` and `fa` evaluation enable:

```text
USE_DRIVABLE_GUIDANCE=true
DRIVABLE_SDF_SOURCE=data_map
```

### 6.2 Context construction

```text
run_pdm_score_feaxdrive.py
-> NavSimScenario(scene, map_root, map_version)
-> map_api
-> build_local_drivable_sdf_context_from_map_api()
-> agent.set_drivable_sdf_context(ctx)
```

### 6.3 Planner usage

During sampling:

```text
x_recon = _apply_drivable_guidance(x_recon, index)
```

The guidance projects or adjusts the predicted trajectory toward the drivable region using local SDF and vehicle footprint constraints.

---

## 7. Scorer isolation

### 7.1 Training reward scorer

Used only by FA-GRPO training:

```text
feaxdrive_fagrpo_scorer.py
feaxdrive_fagrpo_comfort_metrics.py
pdm_score_feaxdrive_fagrpo.py
```

### 7.2 Final evaluation scorer

Used by benchmark evaluation:

```text
pdm_scorer.py
pdm_comfort_metrics.py
pdm_score.py
```

Do not replace the standard scorer with the FA-GRPO scorer. Reward shaping and benchmark evaluation must remain separate.

---

## 8. What to keep and what to remove

Keep:

```text
navsim/
scripts/repro/slurm/
internvl_chat/
docs/
download/
utils/
vqa_evaluation/
README.md
requirements.txt
environment.yml
setup.py
```

Remove from code release:

```text
backup/
source_snapshots/
navsim/exp/
__pycache__/
*.pyc
*.ckpt
*.pt
*.pth
*.safetensors
*.pkl
*.pickle
*.gz
*.tar.gz
*.zip
```

Do not aggressively delete `recogdrive/` or compatibility wrappers until an import-graph analysis confirms they are unused by all configs and checkpoints.

---

## 9. Development recommendations

### 9.1 New reward

Add isolated files:

```text
feaxdrive_xxx_scorer.py
feaxdrive_xxx_comfort_metrics.py
pdm_score_feaxdrive_xxx.py
feaxdrive_diffusion_planner_xxx.py
```

Do not edit standard `pdm_scorer.py` for training reward experiments.

### 9.2 New evaluation metric

Add metric columns in `run_pdm_score_feaxdrive.py` without changing standard PDM score.

### 9.3 Driving style or chassis adaptation

Recommended extension points:

```text
FeaXDriveFeatureBuilder
FeaXDriveAgent.forward()
FeaXDriveDiffusionPlanner conditioning
TrajectoryProjector
run_pdm_score_feaxdrive.py optional metrics
```

Avoid modifying:

```text
pdm_scorer.py
pdm_comfort_metrics.py
```

unless intentionally changing the benchmark definition.

---

## 10. Minimal sanity checks

```bash
bash -n scripts/repro/slurm/submit_cache_feaxdrive.slurm
bash -n scripts/repro/slurm/submit_train_feaxdrive.slurm
bash -n scripts/repro/slurm/submit_eval_feaxdrive.slurm

python -m py_compile \
  navsim/planning/script/run_dataset_caching_multi_node.py \
  navsim/planning/script/run_training_feaxdrive_rl.py \
  navsim/planning/script/run_pdm_score_feaxdrive.py \
  navsim/agents/feaxdrive/feaxdrive_agent_entry_v4drive.py \
  navsim/agents/feaxdrive/feaxdrive_diffusion_planner.py \
  navsim/agents/feaxdrive/feaxdrive_diffusion_planner_fagrpo.py
```
