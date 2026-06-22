#!/bin/bash
#SBATCH -J metriccache1
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/CVLA/navsim/exp/logs/%x-%j.out
#SBATCH --error=/path/to/CVLA/navsim/exp/logs/%x-%j.err

set -euo pipefail

mkdir -p /path/to/CVLA/navsim/exp/logs

bash /path/to/CVLA/scripts/cache_dataset/run_metric_caching_exp.sh