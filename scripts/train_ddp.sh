#!/bin/bash
# DDP Training for SAFE-Det (CCPE / FireSight)
# Usage: bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 0,1,2,3

set -e

CONFIG=${1:-configs/ccpe_single_1024.yaml}
GPUS=${2:-0,1,2,3}

# Count number of GPUs from comma-separated list
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Select specific GPUs
export CUDA_VISIBLE_DEVICES=${GPUS}

# NCCL settings for A100 PCIe
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

echo "=========================================="
echo " SAFE-Det Training"
echo " Config: ${CONFIG}"
echo " GPUs: ${GPUS} (${NUM_GPUS} devices)"
echo "=========================================="

# Activate conda env
CONDA_PREFIX=${CONDA_PREFIX:-/home/whamidouche/ssdprivate/conda_envs/fire-smoke-rtdetrv4}

cd /home/whamidouche/ssdprivate/fire-smoke-ccpe

${CONDA_PREFIX}/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    --config ${CONFIG}
