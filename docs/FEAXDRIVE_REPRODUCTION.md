# FeaXDrive Reproduction Guide

This document is the main command-oriented guide for reproducing the current FeaXDrive cache, training, and evaluation chain.

---

## 1. Mode definitions

### Training modes

| MODE | Meaning |
|---|---|
| `eps` / `orin` | epsilon-pred imitation learning baseline |
| `dt` | x0-pred / trajectory-centric imitation learning |
| `dyn` | x0-pred plus dynamics feasibility training |
| `fa` / `fa_grpo` / `fagrpo` | FA-GRPO reinforcement learning fine-tuning |

There is no `drivedyn` training mode.

### Evaluation modes

| MODE | Meaning |
|---|---|
| `eps` / `orin` | epsilon-pred checkpoint evaluation |
| `dt` | x0-pred checkpoint evaluation |
| `dyn` | dyn checkpoint evaluation without drivable guidance |
| `drivedyn` | dyn checkpoint + data-map drivable guidance |
| `fa` / `fa_grpo` / `fagrpo` | FA-GRPO checkpoint + data-map drivable guidance |

---

## 2. Cache

Unified script:

```bash
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

Supported cache modes:

```text
CACHE_MODE=check
CACHE_MODE=hidden_train
CACHE_MODE=metric_train
CACHE_MODE=metric_eval
CACHE_MODE=all_train
CACHE_MODE=all_eval
CACHE_MODE=all
```

### 2.1 Hidden-state cache

Hidden-state cache must use:

```text
run_dataset_caching_multi_node.py
```

This is required because the script constructs `SceneLoader(..., load_image_path=True)`. If hidden cache is generated through the training entry, image arrays may be converted into invalid long string paths.

#### navtrain hidden cache

```bash
cd /path/to/FeaXDrive

HIDDEN_CACHE=/path/to/navcache/navtrain_hidden_state

sbatch --export=ALL,\
CACHE_MODE=hidden_train,\
HIDDEN_SPLIT=navtrain,\
HIDDEN_CACHE=${HIDDEN_CACHE},\
HIDDEN_NPROC_PER_NODE=1,\
FORCE_HIDDEN_CACHE=false \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

#### Impromptu long-tail hidden cache

```bash
cd /path/to/FeaXDrive

IMP_HIDDEN_CACHE=/path/to/data/navcache/imp_hidden_cache_orin

sbatch --export=ALL,\
CACHE_MODE=hidden_train,\
HIDDEN_SPLIT=navlt_imp_trainval,\
HIDDEN_CACHE=${IMP_HIDDEN_CACHE},\
HIDDEN_NPROC_PER_NODE=1,\
FORCE_HIDDEN_CACHE=false \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

Check:

```bash
du -sh ${IMP_HIDDEN_CACHE}
find ${IMP_HIDDEN_CACHE} -name "internvl_feature.gz" | wc -l
find ${IMP_HIDDEN_CACHE} -name "trajectory_target.gz" | wc -l
```

### 2.2 Metric cache

#### FA-GRPO training metric cache

```bash
cd /path/to/FeaXDrive

IMP_METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache

sbatch --export=ALL,\
CACHE_MODE=metric_train,\
TRAIN_METRIC_SPLIT=navlt_imp_trainval,\
METRIC_CACHE_TRAIN=${IMP_METRIC_CACHE} \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

#### Evaluation metric cache

```bash
cd /path/to/FeaXDrive

METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
CACHE_MODE=metric_eval,\
EVAL_SPLIT=navtest,\
METRIC_CACHE_EVAL=${METRIC_CACHE} \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

---

## 3. Training

Unified script:

```bash
scripts/repro/slurm/submit_train_feaxdrive.slurm
```

### 3.1 eps / orin training

```bash
cd /path/to/FeaXDrive

HIDDEN_CACHE=/path/to/navcache/navtrain_hidden_state

sbatch --export=ALL,\
MODE=eps,\
SPLIT=navtrain,\
EXP_GROUP=navlt_eps,\
EXP_NAME=training_recogdrive_agent_eps,\
NPROC_PER_NODE=4,\
MAX_EPOCHS=100,\
BATCH_SIZE=32,\
USE_CACHE_WITHOUT_DATASET=true,\
HIDDEN_CACHE=${HIDDEN_CACHE} \
scripts/repro/slurm/submit_train_feaxdrive.slurm
```

### 3.2 dt / x0 training

```bash
cd /path/to/FeaXDrive

HIDDEN_CACHE=/path/to/navcache/navtrain_hidden_state

sbatch --export=ALL,\
MODE=dt,\
SPLIT=navtrain,\
EXP_GROUP=navlt_x0,\
EXP_NAME=training_recogdrive_agent_x0,\
NPROC_PER_NODE=4,\
MAX_EPOCHS=100,\
BATCH_SIZE=32,\
USE_CACHE_WITHOUT_DATASET=true,\
HIDDEN_CACHE=${HIDDEN_CACHE} \
scripts/repro/slurm/submit_train_feaxdrive.slurm
```

### 3.3 dyn training

```bash
cd /path/to/FeaXDrive

HIDDEN_CACHE=/path/to/navcache/navtrain_hidden_state

sbatch --export=ALL,\
MODE=dyn,\
SPLIT=navtrain,\
EXP_GROUP=navlt_dyn,\
EXP_NAME=training_recogdrive_agent_dyn,\
NPROC_PER_NODE=4,\
MAX_EPOCHS=100,\
BATCH_SIZE=32,\
USE_CACHE_WITHOUT_DATASET=true,\
HIDDEN_CACHE=${HIDDEN_CACHE} \
scripts/repro/slurm/submit_train_feaxdrive.slurm
```

Verified dyn checkpoint used for FA-GRPO initialization:

```bash
CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt
```

### 3.4 FA-GRPO training

Default reward weights:

```text
progress_weight = 10.0
ttc_weight = 5.0
comfortable_weight = 10.0
gamma_denoising = 0.6
bc_coeff = 0.1
```

Command:

```bash
cd /path/to/FeaXDrive

CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt

IMP_HIDDEN_CACHE=/path/to/data/navcache/imp_hidden_cache_orin

IMP_METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache

sbatch --export=ALL,\
MODE=fa,\
SPLIT=navlt_imp_trainval,\
EXP_GROUP=navlt_imp_fa_grpo,\
EXP_NAME=training_feaxdrive_fa_grpo_imp,\
NPROC_PER_NODE=4,\
MAX_EPOCHS=1,\
FA_MAX_EPOCHS=1,\
BATCH_SIZE=8,\
FA_BATCH_SIZE=8,\
USE_CACHE_WITHOUT_DATASET=true,\
FORCE_CACHE_COMPUTATION=false,\
HIDDEN_CACHE=${IMP_HIDDEN_CACHE},\
METRIC_CACHE=${IMP_METRIC_CACHE},\
CKPT_FEAX_IL=${CKPT_DYN},\
INIT_CKPT=${CKPT_DYN} \
scripts/repro/slurm/submit_train_feaxdrive.slurm
```

Find checkpoint:

```bash
find navsim/exp/navlt_imp_fa_grpo/training_feaxdrive_fa_grpo_imp \
  -name "*.ckpt" -printf "%T@ %p\n" | sort -n | tail -20
```

---

## 4. Evaluation

Unified script:

```bash
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

### 4.1 dyn evaluation

```bash
cd /path/to/FeaXDrive

CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt
METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
MODE=dyn,\
SPLIT=navlt_imp_pure_test,\
BUILD_METRIC_CACHE=0,\
METRIC_CACHE=${METRIC_CACHE},\
CKPT_FEAX_DYN=${CKPT_DYN} \
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

### 4.2 drivedyn evaluation

```bash
cd /path/to/FeaXDrive

CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt
METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
MODE=drivedyn,\
SPLIT=navlt_imp_pure_test,\
BUILD_METRIC_CACHE=0,\
METRIC_CACHE=${METRIC_CACHE},\
DRIVABLE_SDF_SOURCE=data_map,\
USE_DRIVABLE_GUIDANCE=true,\
CKPT_FEAX_DYN=${CKPT_DYN} \
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

### 4.3 FA-GRPO evaluation

```bash
cd /path/to/FeaXDrive

CKPT_FA=/path/to/FeaXDrive/navsim/exp/navlt_imp_fa_grpo/training_feaxdrive_fa_grpo_imp/2026.06.20.14.56.02/lightning_logs/version_1677712/checkpoints/0.ckpt

CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt

METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
MODE=fa_grpo,\
SPLIT=navlt_imp_pure_test,\
BUILD_METRIC_CACHE=0,\
METRIC_CACHE=${METRIC_CACHE},\
DRIVABLE_SDF_SOURCE=data_map,\
USE_DRIVABLE_GUIDANCE=true,\
CKPT_FEAX_FA_GRPO=${CKPT_FA},\
CKPT_FEAX_IL=${CKPT_DYN},\
CKPT_FEAX_DYN=${CKPT_DYN} \
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

---

## 5. Result lookup

```bash
cd /path/to/FeaXDrive

find navsim/exp -type f \( -name "*.csv" -o -name "*.txt" -o -name "*.json" \) \
  -printf "%T@ %p\n" | sort -n | tail -50
```

Latest Slurm log:

```bash
LOG=$(ls -t navsim/exp/logs/*.out 2>/dev/null | head -1)
tail -f ${LOG}
```

---

## 6. Common failure cases

### Hidden cache reports `File name too long`

Use the unified cache script and source-style cache entry. Do not generate hidden cache via the training script.

### FA-GRPO reports `KeyError: token`

Hidden cache and metric cache are not aligned. Use:

```text
HIDDEN_SPLIT=navlt_imp_trainval
TRAIN_METRIC_SPLIT=navlt_imp_trainval
USE_CACHE_WITHOUT_DATASET=true
```

### FA-GRPO training completes without checkpoint

Check that `run_training_feaxdrive_rl.py` contains an unmonitored checkpoint callback:

```bash
grep -n "checkpoint_callbacks\|filename=\"epoch" -A30 \
  navsim/planning/script/run_training_feaxdrive_rl.py
```

### Eval accidentally uses FA-GRPO scorer

Final eval should use standard:

```text
pdm_scorer.py
pdm_comfort_metrics.py
```

FA-GRPO scorer is only for training reward.
