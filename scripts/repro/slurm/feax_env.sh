#!/usr/bin/env bash
FEAX_SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${FEAX_SLURM_DIR}/../../envs/load_local_env.sh"
feax_activate_conda
