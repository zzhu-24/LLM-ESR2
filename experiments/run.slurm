#!/usr/bin/env bash
#SBATCH --job-name=efficient_compress
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=your@1993604693@qq.com
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

DATASET="${DATASET:-musical}"
SEEDS="${SEEDS:-42}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-llmesr}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/home/zhuzhehua/2025/LLM-ESR2}}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"

mkdir -p "slurm_results/${DATASET}"

echo "Running efficient compression experiment"
echo "repo: ${REPO_DIR}"
echo "dataset: ${DATASET}"
echo "seeds: ${SEEDS}"
echo "gpu_id: ${GPU_ID}"

export SEEDS
export GPU_ID

bash experiments/efficient_compress.bash "${DATASET}" 2>&1 | tee "slurm_results/${DATASET}/efficient_compress_${SLURM_JOB_ID:-manual}.log"
