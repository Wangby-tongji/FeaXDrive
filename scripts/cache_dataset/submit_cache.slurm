#!/bin/bash
#SBATCH -J recog_cache_8gpu
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:8
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/CVLA/navsim/exp/logs/%x-%j.out
#SBATCH --error=/path/to/CVLA/navsim/exp/logs/%x-%j.err

set -euo pipefail

mkdir -p /path/to/CVLA/navsim/exp/logs
export RUN_ID="${SLURM_JOB_ID}"

bash /path/to/CVLA/scripts/cache_dataset/run_caching_recogdrive_hidden_state_exp.sh