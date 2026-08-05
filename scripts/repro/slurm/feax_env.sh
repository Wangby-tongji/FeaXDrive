#!/usr/bin/env bash
# Shared local-environment setup for the public FeaXDrive Slurm entrypoints.
# Users may create scripts/envs/env_local.sh from env_template.sh; it is ignored
# by Git and may contain machine-specific data, model, cache, and output paths.

FEAX_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FEAX_DEFAULT_ROOT="$(cd -- "${FEAX_SLURM_DIR}/../../.." && pwd)"
FEAX_LOCAL_ENV="${FEAX_DEFAULT_ROOT}/scripts/envs/env_local.sh"

if [ -f "${FEAX_LOCAL_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${FEAX_LOCAL_ENV}"
fi

export FEAX_ROOT="${FEAX_ROOT:-${FEAXDRIVE_ROOT:-${FEAX_DEFAULT_ROOT}}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${FEAX_ROOT}}"
export NAVSIM_EXP_BASE="${NAVSIM_EXP_BASE:-${NAVSIM_EXP_ROOT:-${FEAX_ROOT}/navsim/exp}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${NAVSIM_EXP_BASE}}"
export PYTHONPATH="${FEAX_ROOT}:${PYTHONPATH:-}"

feax_require_dir() {
  local variable_name="$1"
  local value="${!variable_name:-}"
  if [ -z "${value}" ]; then
    echo "ERROR: ${variable_name} is unset. Copy scripts/envs/env_template.sh to scripts/envs/env_local.sh and configure it." >&2
    exit 2
  fi
  if [ ! -d "${value}" ]; then
    echo "ERROR: ${variable_name} is not a directory: ${value}" >&2
    exit 2
  fi
}

CONDA_ENV="${CONDA_ENV:-feaxdrive}"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Initialize Conda before submitting this job." >&2
  return 1
fi

# Load Conda's shell function in non-interactive Slurm shells.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
