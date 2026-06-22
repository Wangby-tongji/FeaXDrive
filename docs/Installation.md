# Installation

This document describes the current FeaXDrive environment setup. It keeps only the parts needed for the verified NAVSIM reproduction path and optional VLM SFT.

---

## 1. Basic environment

Clone or enter the project:

```bash
cd /path/to/FeaXDrive
```

Create or activate the environment used for NAVSIM / FeaXDrive:

```bash
conda activate navsim-wby
```

Install the project in editable mode:

```bash
pip install -e .
```

If dependencies are missing:

```bash
pip install -r requirements.txt
```

---

## 2. Optional InternVL / VLM SFT dependencies

Only install these if you need to fine-tune the VLM or run the InternVL training tools:

```bash
pip install -r internvl_chat/internvl_chat.txt
```

For the verified FeaXDrive planner reproduction, the usual path is to use the pretrained ReCogDrive VLM checkpoint and cache hidden states. Full VLM SFT is optional.

---

## 3. Required paths

Set these paths before cache / train / eval:

```bash
export FEAX_ROOT=/path/to/FeaXDrive
export NAVSIM_DEVKIT_ROOT=${FEAX_ROOT}
export NAVSIM_EXP_ROOT=${FEAX_ROOT}/navsim/exp
export PYTHONPATH=${FEAX_ROOT}:${PYTHONPATH:-}

export OPENSCENE_DATA_ROOT=/path/to/navdata
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/path/to/navdata/maps/nuplan-maps-v1.0
export VLM_CKPT=/path/to/ReCogDrive-VLM-2B
```

Quick check:

```bash
test -d "${FEAX_ROOT}"
test -d "${OPENSCENE_DATA_ROOT}"
test -d "${NUPLAN_MAPS_ROOT}"
test -d "${VLM_CKPT}"
```

---

## 4. Slurm script sanity check

```bash
cd /path/to/FeaXDrive

bash -n scripts/repro/slurm/submit_cache_feaxdrive.slurm
bash -n scripts/repro/slurm/submit_train_feaxdrive.slurm
bash -n scripts/repro/slurm/submit_eval_feaxdrive.slurm
```

---

## 5. Notes

- Metric caching should use NumPy `>=1.26.4` to avoid reproducibility issues in metric-cache generation.
- Do not generate hidden-state cache through the training entrypoint.
- Do not restore or rely on `use_constraint_projection`; it is deprecated in the current FeaXDrive chain.
