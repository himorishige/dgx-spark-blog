# R1: DGX Spark でロボットアームの強化学習を試してみた（Isaac Sim + Isaac Lab + SO-ARM101）

Isaac Sim + Isaac Lab 環境で SO-ARM101 ロボットアームの Reaching タスクにカスタム報酬を設計し、ベースラインとの比較実験を行ったスクリプトです。

## Results

64 並列環境、1000 イテレーション（約 9 分 / パターン）での比較結果です。

| Configuration | position error | Mean Reward | Throughput |
|---------------|---------------|-------------|------------|
| Baseline | 0.0987 | 0.23 | 3,514 steps/s |
| Custom A（パラメータ調整） | 0.1279 | -0.72 | 3,446 steps/s |
| Custom B（関節リミット回避追加） | **0.0802** | 0.03 | 3,043 steps/s |

Custom A はペナルティ強化のみで精度が悪化。Custom B は関節リミット回避報酬を追加したことで正則化効果が働き、ベースラインを上回りました。

## Environment

- **Hardware**: NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory)
- **Isaac Sim**: 5.1.0-rc.19 (source build, aarch64)
- **Isaac Lab**: 0.54.3
- **isaac_so_arm101**: v1.2.0
- **RL Framework**: RSL-RL (PPO)

## Scripts

| File | Description |
|------|-------------|
| `scripts/r1-custom-reach-env.py` | Custom reward configs (A: parameter tuning, B: joint limit avoidance) + Gymnasium registration |
| `scripts/r1-run-comparison.sh` | Run all 3 configurations sequentially for comparison |

### r1-custom-reach-env.py

2 つのカスタム環境を定義しています。

| Env ID | Changes from Baseline |
|--------|----------------------|
| `Isaac-SO-ARM101-Reach-CustomA-v0` | position penalty -0.2→-0.5, tanh std 0.1→0.05, action_rate -0.0001→-0.001 |
| `Isaac-SO-ARM101-Reach-CustomB-v0` | Custom A + joint limit avoidance reward (weight=0.05, std=0.2) |

`joint_pos_limit_avoidance` は各関節が可動域リミットから離れているほど高い報酬を返す関数です。tanh カーネルを使い、リミットから 0.2 rad（約 11 度）以内で報酬が急落する設計になっています。

## Quick Start

```bash
# Prerequisites: Isaac Sim (source build) + Isaac Lab + isaac_so_arm101 installed
# See article for DGX Spark aarch64 build instructions

cd ~/works/robotics/IsaacLab

# Required for DGX Spark
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"

# 1. Copy custom env to isaac_so_arm101 package
cp scripts/r1-custom-reach-env.py \
   ~/works/robotics/isaac_so_arm101/src/isaac_so_arm101/

# 2. Register custom envs (add import to __init__.py)
echo "import isaac_so_arm101.r1_custom_reach_env" >> \
   ~/works/robotics/isaac_so_arm101/src/isaac_so_arm101/__init__.py

# 3. Train baseline
./isaaclab.sh -p ~/works/robotics/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Reach-v0 \
    --headless --num_envs 64 --max_iterations 1000

# 4. Train Custom B (joint limit avoidance)
./isaaclab.sh -p ~/works/robotics/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Reach-CustomB-v0 \
    --headless --num_envs 64 --max_iterations 1000

# 5. Play learned policy
./isaaclab.sh -p ~/works/robotics/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Reach-Play-v0 \
    --num_envs 4 --video --video_length 200

# 6. Compare in TensorBoard
tensorboard --logdir logs/
```

## DGX Spark aarch64 Gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| No Isaac Sim binary | Official builds are x86_64 only | Source build (~13 min) |
| Build fails | GCC 13 incompatibility | Install GCC 11 |
| Crash (OpenMP) | libgomp linking issue | `LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"` |
| Hang via SSH | `XOpenDisplay` called even in headless | Set DISPLAY/XAUTHORITY or use desktop |
| `uv sync` fails | Source build deps not on PyPI | `pip install --no-deps` |

## References

- [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)
- [Isaac Sim](https://github.com/isaac-sim/IsaacSim)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [DGX Spark Isaac Playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/isaac)
