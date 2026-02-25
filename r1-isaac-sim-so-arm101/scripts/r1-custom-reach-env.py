# Copyright (c) 2026, Hiroshi Morishige
# SPDX-License-Identifier: MIT
#
# Custom reward configurations for SO-ARM101 Reaching task.
# Used with isaac_so_arm101 (https://github.com/MuammerBay/isaac_so_arm101)
#
# Usage:
#   Place this file under isaac_so_arm101/src/isaac_so_arm101/ and
#   register the Gymnasium envs (see __init__.py snippet at bottom).
#   Then run:
#     ./isaaclab.sh -p .../train.py --task Isaac-SO-ARM101-Reach-CustomA-v0
#     ./isaaclab.sh -p .../train.py --task Isaac-SO-ARM101-Reach-CustomB-v0

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaac_so_arm101.reach.config.joint_pos_env_cfg import (
    SoArm101ReachEnvCfg,
)
from isaac_so_arm101.reach.reach_env_cfg import RewardsCfg

# ---------------------------------------------------------------------------
# Custom reward function
# ---------------------------------------------------------------------------


def joint_pos_limit_avoidance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward that encourages joints to stay away from position limits.

    Uses a tanh kernel: returns ~1.0 when far from limits, ~0.0 near limits.
    With std=0.2, the reward drops sharply within 0.2 rad (~11 deg) of a limit.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    soft_limits = asset.data.soft_joint_pos_limits

    # Distance to nearest limit for each joint
    dist_to_lower = joint_pos - soft_limits[..., 0]
    dist_to_upper = soft_limits[..., 1] - joint_pos
    dist_to_nearest = torch.minimum(dist_to_lower, dist_to_upper)

    # tanh kernel: far from limit -> ~1.0, near limit -> ~0.0
    return torch.mean(torch.tanh(dist_to_nearest / std), dim=-1)


# ---------------------------------------------------------------------------
# Custom A: Parameter tuning only (no new reward functions)
# ---------------------------------------------------------------------------


@configclass
class SoArm101ReachCustomACfg(SoArm101ReachEnvCfg):
    """Reach env with tuned reward weights.

    Changes from baseline:
      - position tracking penalty: -0.2 -> -0.5
      - fine_grained tanh std: 0.1 -> 0.05
      - action_rate penalty: -0.0001 -> -0.001
    """

    def __post_init__(self):
        super().__post_init__()

        # Stronger distance penalty (-0.2 -> -0.5)
        self.rewards.end_effector_position_tracking.weight = -0.5
        # Tighter tanh sensitivity (std 0.1 -> 0.05)
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.05
        # Require some smoothness from the start (-0.0001 -> -0.001)
        self.rewards.action_rate.weight = -0.001


# ---------------------------------------------------------------------------
# Custom B: Parameter tuning + joint limit avoidance reward
# ---------------------------------------------------------------------------


@configclass
class CustomBRewardsCfg(RewardsCfg):
    """Baseline rewards plus joint limit avoidance."""

    joint_limit_avoidance: RewTerm = RewTerm(
        func=joint_pos_limit_avoidance,
        weight=0.05,
        params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class SoArm101ReachCustomBCfg(SoArm101ReachEnvCfg):
    """Reach env with tuned weights + joint limit avoidance reward.

    Same parameter changes as Custom A, plus a new reward term that
    penalizes joints near their position limits (acts as regularization).
    """

    rewards: CustomBRewardsCfg = CustomBRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Same tuning as Custom A
        self.rewards.end_effector_position_tracking.weight = -0.5
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.05
        self.rewards.action_rate.weight = -0.001


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------
# Add the following to isaac_so_arm101/__init__.py (or run this file directly)
# to register custom envs alongside the baseline.

gym.register(
    id="Isaac-SO-ARM101-Reach-CustomA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:SoArm101ReachCustomACfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Reach-CustomB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:SoArm101ReachCustomBCfg",
    },
    disable_env_checker=True,
)
