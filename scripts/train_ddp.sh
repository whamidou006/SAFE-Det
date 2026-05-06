#!/bin/bash
# DDP Training for SAFE-Det (CCPE / FireSight)
# Usage:
#   bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 0,1
#   bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 0,1 --batch-size 16 --lr 4e-4
# Any args after <gpus> are forwarded verbatim to train.py.

set -e

CONFIG=${1:-configs/ccpe_single_1024.yaml}
GPUS=${2:-0,1,2,3}
shift $(( $# >= 2 ? 2 : $# ))   # drop $1 and $2 if present, leave the rest as $@

# Count number of GPUs from comma-separated list
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Select specific GPUs
export CUDA_VISIBLE_DEVICES=${GPUS}

# NCCL settings for A100 PCIe
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

# CUDA allocator: expandable_segments avoids fragmentation OOMs that
# show up when set_to_none=True churns gradient buffers each step.
# At bs=32 / 1024² we sit at ~65 GB steady-state on H100 NVL (95 GB),
# but peak forward activations spike to >90 GB on busy mosaic batches.
# Without this, the allocator fragments and a 768 MiB Linear matmul
# can fail to find a contiguous block. (Per the OOM error message:
# "If reserved but unallocated memory is large try setting
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid
# fragmentation.")
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=========================================="
echo " SAFE-Det Training"
echo " Config: ${CONFIG}"
echo " GPUs: ${GPUS} (${NUM_GPUS} devices)"
if [ $# -gt 0 ]; then
  echo " Extra args: $@"
fi
echo "=========================================="

# Activate conda env (override with $CONDA_PREFIX if your env lives elsewhere).
CONDA_PREFIX=${CONDA_PREFIX:-/home/whamidouche/ssdprivate/conda_envs/condor-bench}

cd "$(dirname "$0")/.."

${CONDA_PREFIX}/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    --config ${CONFIG} \
    "$@"
