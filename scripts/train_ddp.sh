#!/bin/bash
# DDP Training for CCPE Fire/Smoke Detection
# Usage: bash scripts/train_ddp.sh configs/ccpe_single_1024.yaml 4

set -e

CONFIG=${1:-configs/ccpe_single_1024.yaml}
NUM_GPUS=${2:-4}

# NCCL settings for our hardware (A100 PCIe)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

echo "=========================================="
echo " CCPE Fire/Smoke Training"
echo " Config: ${CONFIG}"
echo " GPUs: ${NUM_GPUS}"
echo "=========================================="

# Activate conda env
CONDA_PREFIX=${CONDA_PREFIX:-/home/whamidouche/ssdprivate/conda_envs/fire-smoke-rtdetrv4}

cd /home/whamidouche/ssdprivate/fire-smoke-ccpe

${CONDA_PREFIX}/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train.py \
    --config ${CONFIG}
