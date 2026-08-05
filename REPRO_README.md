# FeaXDrive 全流程 README

本文档用于复现当前已经跑通的 FeaXDrive 链路，重点是命令、路径、配置和实验调度。当前复现闭环包括：

- hidden-state cache
- metric cache
- orin / eps、dt、dyn、FA-GRPO 训练
- orin / eps、dt、dyn、drivedyn、FA-GRPO 推理评测
- FA-GRPO reward scorer 与最终标准 eval scorer 隔离

> 重要约定：`drivedyn` 不是训练模式。`drivedyn = dyn checkpoint + eval-time data_map drivable guidance`。  
> 当前官方链路不再使用 `use_constraint_projection`，不要恢复或依赖该参数。

---

## 1. 项目路径与环境变量

默认项目路径：

```bash
FEAX_ROOT=/path/to/FeaXDrive
cd ${FEAX_ROOT}
```

默认数据与模型路径：

```bash
export FEAX_ROOT=/path/to/FeaXDrive
export NAVSIM_DEVKIT_ROOT=${FEAX_ROOT}
export NAVSIM_EXP_BASE=${FEAX_ROOT}/navsim/exp
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_BASE}
export PYTHONPATH=${FEAX_ROOT}:${PYTHONPATH:-}

export OPENSCENE_DATA_ROOT=/path/to/navdata
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/path/to/navdata/maps/nuplan-maps-v1.0
export VLM_CKPT=/path/to/ReCogDrive-VLM-2B
```

进入环境：

```bash
source ~/.bashrc
conda activate feaxdrive
```

快速检查：

```bash
test -d ${FEAX_ROOT}
test -d ${OPENSCENE_DATA_ROOT}
test -d ${NUPLAN_MAPS_ROOT}
test -d ${VLM_CKPT}

bash -n scripts/repro/slurm/submit_cache_feaxdrive.slurm
bash -n scripts/repro/slurm/submit_train_feaxdrive.slurm
bash -n scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

---

## 2. 模式定义

### 2.1 训练模式

| MODE | 含义 | checkpoint 初始化 | guidance |
|---|---|---|---|
| `orin` / `eps` | epsilon-pred baseline | 默认从头训练 | 无 |
| `dt` | x0-pred / trajectory-centric baseline | 默认从头训练 | 无 |
| `dyn` | x0-pred + dynamics feasibility training | 默认从头训练 | 无 |
| `fa_grpo` / `fagrpo` / `fa` | FA-GRPO RL fine-tuning | 基于 dyn checkpoint | 训练 reward 用 FA-GRPO scorer |

无 `drivedyn` 训练模式。

### 2.2 推理 / 评测模式

| MODE | 含义 |
|---|---|
| `orin` / `eps` | epsilon-pred checkpoint，无 drivable guidance |
| `dt` | x0 checkpoint，无 drivable guidance |
| `dyn` | dyn checkpoint，无 drivable guidance |
| `drivedyn` | dyn checkpoint + `data_map` drivable guidance |
| `fa_grpo` / `fagrpo` / `fa` | FA-GRPO checkpoint + `data_map` drivable guidance |

---

## 3. Cache 复现

统一 cache 调度脚本：

```bash
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

支持：

```text
CACHE_MODE=check
CACHE_MODE=hidden_train
CACHE_MODE=metric_train
CACHE_MODE=metric_eval
CACHE_MODE=all_train
CACHE_MODE=all_eval
CACHE_MODE=all
```

### 3.1 Hidden-state cache

hidden-state cache 必须走源码 cache 入口：

```text
run_dataset_caching_multi_node.py
-> SceneLoader(..., load_image_path=True)
-> Dataset.cache_dataset()
-> FeaXDriveFeatureBuilder.compute_features()
-> internvl_feature.gz / trajectory_target.gz
```

不要用 `run_training_feaxdrive_rl.py` 生成 hidden cache，否则可能把 ndarray 图像当成文件名，出现：

```text
OSError: [Errno 36] File name too long: '[[[156 199 231] ...]]'
```

#### 3.1.1 生成 navtrain hidden cache

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

#### 3.1.2 生成 Impromptu long-tail hidden cache

当前复现 FA-GRPO 时使用的 imp hidden cache：

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

完成后检查：

```bash
du -sh ${IMP_HIDDEN_CACHE}
find ${IMP_HIDDEN_CACHE} -name "internvl_feature.gz" | wc -l
find ${IMP_HIDDEN_CACHE} -name "trajectory_target.gz" | wc -l
find ${IMP_HIDDEN_CACHE} -type f | head -20
```

### 3.2 Metric cache

metric cache 有两类：

| 用途 | 推荐路径 | 说明 |
|---|---|---|
| FA-GRPO 训练 reward | `${FEAX_ROOT}/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache` | 与 `navlt_imp_trainval` 对齐 |
| 最终 eval | `${FEAX_ROOT}/navsim/exp/metric_cache_navtest` | 可用于 `navlt_imp_pure_test` 测试时复用 navtest metric cache |

#### 3.2.1 生成 FA-GRPO 训练 metric cache

```bash
cd /path/to/FeaXDrive

IMP_METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache

sbatch --export=ALL,\
CACHE_MODE=metric_train,\
TRAIN_METRIC_SPLIT=navlt_imp_trainval,\
METRIC_CACHE_TRAIN=${IMP_METRIC_CACHE} \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

#### 3.2.2 生成 eval metric cache

```bash
cd /path/to/FeaXDrive

METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
CACHE_MODE=metric_eval,\
EVAL_SPLIT=navtest,\
METRIC_CACHE_EVAL=${METRIC_CACHE} \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

#### 3.2.3 一次性生成 train cache

```bash
cd /path/to/FeaXDrive

IMP_HIDDEN_CACHE=/path/to/data/navcache/imp_hidden_cache_orin
IMP_METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache

sbatch --export=ALL,\
CACHE_MODE=all_train,\
HIDDEN_SPLIT=navlt_imp_trainval,\
TRAIN_METRIC_SPLIT=navlt_imp_trainval,\
HIDDEN_CACHE=${IMP_HIDDEN_CACHE},\
METRIC_CACHE_TRAIN=${IMP_METRIC_CACHE},\
HIDDEN_NPROC_PER_NODE=1,\
FORCE_HIDDEN_CACHE=false \
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

---

## 4. 训练复现

统一训练脚本：

```bash
scripts/repro/slurm/submit_train_feaxdrive.slurm
```

关键训练参数：

```bash
MODE=eps | dt | dyn | fa
SPLIT=navtrain | navlt_imp_trainval
HIDDEN_CACHE=/path/to/hidden_cache
USE_CACHE_WITHOUT_DATASET=true
NPROC_PER_NODE=4
BATCH_SIZE=...
MAX_EPOCHS=...
```

当前训练入口已经支持每 epoch 保存 checkpoint：

```text
ModelCheckpoint(monitor="val/loss_epoch", save_top_k=2, save_last=True)
ModelCheckpoint(monitor=None, save_top_k=-1, every_n_epochs=1, save_last=True)
```

因此 FA-GRPO 只训练 1 epoch 时也会保存：

```text
epoch-000.ckpt
last.ckpt
```

### 4.1 orin / eps 训练

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

### 4.2 dt / x0 训练

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

### 4.3 dyn 训练

`dyn` 是 x0-pred + dynamics feasibility regularization。当前链路不使用 `use_constraint_projection`，而是通过 `lambda_dyn` 和 projector 中的 curvature / lateral acceleration 约束进行训练期正则。

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

当前已验证可作为 FA-GRPO 初始化 / reference 的 dyn checkpoint：

```bash
CKPT_DYN=/path/to/FeaXDrive/navsim/exp/navlt_dyn/training_recogdrive_agent_dyn/2026.03.19.18.05.42/lightning_logs/version_1294653/checkpoints/dyn_101_0.255.ckpt
```

### 4.4 FA-GRPO 训练

FA-GRPO 使用：

```text
agent=feaxdrive_v4drive_fagrpo_agent
planner_variant=fagrpo
grpo=true
reference_policy_checkpoint=dyn checkpoint
metric_cache_path=FA-GRPO training metric cache
```

FA-GRPO reward 默认参数：

```bash
FA_REWARD_PROGRESS_WEIGHT=10.0
FA_REWARD_TTC_WEIGHT=5.0
FA_REWARD_COMFORTABLE_WEIGHT=10.0
FA_GAMMA_DENOISING=0.6
FA_BC_COEFF=0.1
```

提交命令：

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

如果 OOM：

```bash
BATCH_SIZE=4
FA_BATCH_SIZE=4
```

训练完成后查 ckpt：

```bash
find navsim/exp/navlt_imp_fa_grpo/training_feaxdrive_fa_grpo_imp \
  -name "*.ckpt" -printf "%T@ %p\n" | sort -n | tail -20
```

示例已跑通的 FA-GRPO checkpoint：

```bash
CKPT_FA=/path/to/FeaXDrive/navsim/exp/navlt_imp_fa_grpo/training_feaxdrive_fa_grpo_imp/2026.06.20.14.56.02/lightning_logs/version_1677712/checkpoints/0.ckpt
```

---

## 5. 推理 / 评测复现

统一评测脚本：

```bash
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

关键参数：

```bash
MODE=eps | dt | dyn | drivedyn | fa
SPLIT=navtest | navlt_imp_pure_test
METRIC_CACHE=/path/to/metric_cache
BUILD_METRIC_CACHE=0
DRIVABLE_SDF_SOURCE=data_map
```

最终 eval 使用标准 PDM scorer：

```text
run_pdm_score_feaxdrive.py
-> navsim.evaluate.pdm_score
-> pdm_scorer.py
-> pdm_comfort_metrics.py
```

不会使用 FA-GRPO reward scorer。

### 5.1 orin / eps 推理

```bash
cd /path/to/FeaXDrive

CKPT_EPS=/path/to/best_eps.ckpt
METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
MODE=eps,\
SPLIT=navlt_imp_pure_test,\
BUILD_METRIC_CACHE=0,\
METRIC_CACHE=${METRIC_CACHE},\
CKPT_EPS=${CKPT_EPS} \
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

### 5.2 dt 推理

```bash
cd /path/to/FeaXDrive

CKPT_DT=/path/to/best_x0_100.ckpt
METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/metric_cache_navtest

sbatch --export=ALL,\
MODE=dt,\
SPLIT=navlt_imp_pure_test,\
BUILD_METRIC_CACHE=0,\
METRIC_CACHE=${METRIC_CACHE},\
CKPT_FEAX_DT=${CKPT_DT} \
scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

### 5.3 dyn 推理

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

### 5.4 drivedyn 推理

`drivedyn = dyn checkpoint + data_map drivable guidance`。

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

### 5.5 FA-GRPO 推理

`fa = FA-GRPO checkpoint + data_map drivable guidance + 标准 PDM eval scorer`。

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

## 6. 日志与结果检查

### 6.1 查看任务

```bash
squeue -u $USER
```

### 6.2 查看最新日志

```bash
cd /path/to/FeaXDrive

LOG=$(ls -t navsim/exp/logs/*.out 2>/dev/null | head -1)
tail -f ${LOG}
```

### 6.3 查找评测结果

```bash
find navsim/exp -type f \( -name "*.csv" -o -name "*.txt" -o -name "*.json" \) \
  -printf "%T@ %p\n" | sort -n | tail -50
```

### 6.4 评测日志中需要确认

```text
EVAL MODE=fa
SPLIT=navlt_imp_pure_test
CKPT=.../checkpoints/0.ckpt
GRPO=True
DIFF_TYPE=x0
PLANNER_VARIANT=standard
DRIVABLE_SDF_SOURCE=data_map
USE_DRIVABLE_GUIDANCE=true
METRIC_CACHE=.../metric_cache_navtest
```

对于 FA-GRPO eval，`PLANNER_VARIANT=standard` 是预期行为：评测只加载 FA-GRPO 训练后的 checkpoint，最终打分仍使用标准 eval scorer。

---

## 7. 常见问题

### 7.1 hidden cache 出现 File name too long

原因：错误使用训练入口生成 hidden cache，导致 `cam_f0.image` 是 ndarray 而不是路径。

解决：使用：

```bash
CACHE_MODE=hidden_train
scripts/repro/slurm/submit_cache_feaxdrive.slurm
```

该脚本内部调用：

```text
run_dataset_caching_multi_node.py
```

并确认：

```bash
grep -n "load_image_path=True" navsim/planning/script/run_dataset_caching_multi_node.py
```

### 7.2 FA-GRPO 训练 KeyError token

原因：hidden cache token 集合与 metric cache token 集合不一致。

解决：
- `SPLIT=navlt_imp_trainval`
- `HIDDEN_CACHE=/path/to/data/navcache/imp_hidden_cache_orin`
- `METRIC_CACHE=/path/to/FeaXDrive/navsim/exp/navlt_imp_trainval_fagrpo/metric_cache`
- `USE_CACHE_WITHOUT_DATASET=true`

检查报错 token：

```bash
for t in <token1> <token2>; do
  echo "==== $t hidden ===="
  find ${IMP_HIDDEN_CACHE} -path "*$t*" | head

  echo "==== $t metric ===="
  find ${IMP_METRIC_CACHE} -path "*$t*" | head
done
```

### 7.3 训练完成没有 ckpt

当前版本已经修复。确认：

```bash
grep -n "checkpoint_callbacks\|filename=\"epoch" -A30 \
  navsim/planning/script/run_training_feaxdrive_rl.py
```

应看到：

```text
filename="epoch-{epoch:03d}"
monitor=None
save_last=True
```

---
