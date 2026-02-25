#!/usr/bin/env bash
# Copyright (c) 2026, Hiroshi Morishige
# SPDX-License-Identifier: MIT
#
# Run baseline and custom reward comparisons for SO-ARM101 Reaching task.
# Requires: Isaac Sim + Isaac Lab + isaac_so_arm101 installed.
#
# Usage:
#   cd ~/works/robotics/IsaacLab
#   bash /path/to/r1-run-comparison.sh [--num_envs 64] [--max_iterations 1000]

set -euo pipefail

# --- Configuration ---
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/works/robotics/IsaacLab}"
ISAAC_SO_ARM101_DIR="${ISAAC_SO_ARM101_DIR:-$HOME/works/robotics/isaac_so_arm101}"
TRAIN_SCRIPT="${ISAAC_SO_ARM101_DIR}/src/isaac_so_arm101/scripts/rsl_rl/train.py"
PLAY_SCRIPT="${ISAAC_SO_ARM101_DIR}/src/isaac_so_arm101/scripts/rsl_rl/play.py"

NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"

# --- Environment (DGX Spark aarch64 specific) ---
export LD_PRELOAD="${LD_PRELOAD:-/lib/aarch64-linux-gnu/libgomp.so.1}"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== SO-ARM101 Reach: Reward Comparison ==="
echo "  num_envs: ${NUM_ENVS}"
echo "  max_iterations: ${MAX_ITERATIONS}"
echo ""

cd "${ISAACLAB_DIR}"

# --- 1. Baseline ---
echo "[1/3] Training: Baseline"
./isaaclab.sh -p "${TRAIN_SCRIPT}" \
    --task Isaac-SO-ARM101-Reach-v0 \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}"

echo ""

# --- 2. Custom A (parameter tuning) ---
echo "[2/3] Training: Custom A (parameter tuning)"
./isaaclab.sh -p "${TRAIN_SCRIPT}" \
    --task Isaac-SO-ARM101-Reach-CustomA-v0 \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}"

echo ""

# --- 3. Custom B (parameter tuning + joint limit avoidance) ---
echo "[3/3] Training: Custom B (joint limit avoidance)"
./isaaclab.sh -p "${TRAIN_SCRIPT}" \
    --task Isaac-SO-ARM101-Reach-CustomB-v0 \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}"

echo ""
echo "=== All training complete ==="
echo "Check logs/ directory for TensorBoard results."
echo "  tensorboard --logdir logs/"
echo ""
echo "To play the best model (Custom B):"
echo "  ./isaaclab.sh -p ${PLAY_SCRIPT} \\"
echo "    --task Isaac-SO-ARM101-Reach-Play-v0 \\"
echo "    --num_envs 4 --video --video_length 200"
