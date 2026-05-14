# Reachy Mini Photo Booth Japanese Localization

Reachy Mini Wired + DGX Spark で NVIDIA Spark Reachy Photo Booth Playbook v1.0.1 を
動かし、UI / STT / LLM / TTS / 画像生成プロンプトをフル日本語化する記事用の
スクリプト・検証データを置きます。

公開予定: 2026-05-19 夕方 - 2026-05-20 朝（クラスメソッドフォーラム 2026 デモ展示の前日）

## Phase 1: Reachy Mini Wired Connectivity Smoke Test

`scripts/` 配下の 3 本は、Reachy Mini Wired を DGX Spark に USB-C 接続した直後の
動作確認スクリプトです。`reachy-mini` SDK v1.7.3 を前提にしています。

### Prerequisites

```bash
mise use -g uv@latest python@3.13
sudo usermod -aG dialout $USER     # /dev/ttyACM0 へのアクセス権
# Re-login or reboot to apply group membership
cd ~/works/reachy-mini
uv init --python 3.13
uv add reachy-mini
```

### Scripts

| File | Purpose |
|---|---|
| `scripts/rm1-check-connection.py` | OS device check + SDK import + daemon subprocess 起動 + read-only クライアント接続 + 状態取得 + クリーン切断 |
| `scripts/rm1-step2-head-motion.py` | 相対デルタ +10mm / +10° / +20° の頭部動作（baseline 取得 → Z / pitch / yaw / neutral 復帰 → sleep） |
| `scripts/rm1-step3-expressive-motion.py` | INIT_HEAD_POSE 基準絶対座標で Nod (pitch ±15°) / Shake (yaw ±25°) / Tilt (roll ±15°) / Antennas (±60°) / Final settle の 13 ステップ約 30 秒シーケンス |

### How to run

```bash
cd ~/works/reachy-mini
uv run python scripts/rm1-check-connection.py
uv run python scripts/rm1-step2-head-motion.py
uv run python scripts/rm1-step3-expressive-motion.py
```

実行前に Reachy Mini の頭部上方とアンテナ周辺の物理クリアランスを確保してください。

### Daemon options used

3 本のスクリプトはすべて、共通の daemon 起動パターンを使います:

```
reachy-mini-daemon
  --fastapi-host 127.0.0.1
  --fastapi-port 8765          # :8000 が他プロセスに占有されている環境向け
  --localhost-only
  --no-media                   # GStreamer webrtcsink rust plugin 不要
  --headless
  --wake-up-on-start           # 起動時に自動 wake_up（後述）
  --goto-sleep-on-stop
  --log-level WARNING
```

`--wake-up-on-start` を外すと、クライアント側の `mini.wake_up()` 呼び出しが
実機サーボに届かない事象を確認しました（v1.7.3、daemon 起動時の motor enable
シーケンスとクライアント wake_up が排他になっている可能性）。

### Notes from validation (記事に書く想定の気づき)

1. **port 8000 conflict**: 既存の vLLM / NIM コンテナが `:8000` を握っていると
   daemon 起動失敗。`--fastapi-port 8765` で回避。
2. **`--no-media` required**: GStreamer `webrtcsink` rust plugin 未インストールだと
   `Failed to create webrtcsink element` で daemon が死ぬ。Photo Booth Playbook 側で
   改めて GStreamer 依存を入れることになる。
3. **`--no-wake-up-on-start` の罠**: クライアントの `mini.wake_up()` が
   サーボに反映されない。`--wake-up-on-start`（既定）に倣う。
4. **`get_current_head_pose()` の特性**: IK 解で並進・回転で大きく変わらず、
   姿勢の変化を確認するなら `get_current_joint_positions()` のばらつきを見るのが信頼できる。
5. **`mini.imu` is NoneType**: Wired/Lite 版の制約か `media_backend="none"` の
   副作用か要追跡。

### SDK 内部の主要定数（`reachy_mini.reachy_mini` モジュール内）

| Constant | Value |
|---|---|
| `INIT_HEAD_POSE` | 4x4 identity (xyz = 0, 0, 0) |
| `SLEEP_HEAD_POSE.xyz` | `(-21, 1, -44)` mm |
| `INIT_ANTENNAS_JOINT_POSITIONS` | `[-0.1745, 0.1745]` rad (`±10°`) |
| `SLEEP_ANTENNAS_JOINT_POSITIONS` | `[-3.05, 3.05]` rad (`±174.8°`) |

## Phase 2 以降

Phase 2 では NVIDIA/spark-reachy-photo-booth v1.0.1 (commit `58d24e5`) の
docker compose スタックを DGX Spark で起動し、19 マイクロサービス（Flux NIM /
Parakeet NIM / TRT-LLM / Kokoro / Detectron2 + ByteTrack / NeMo Agent Toolkit /
MinIO / Redpanda / Phoenix 等）を立ち上げます。

その後、日本語化フェーズで以下を差し替えていきます:

- UI: react-i18next で日本語化
- STT: Parakeet 1.1B (en-US) → **ReazonSpeech-NeMo-v2**
- LLM: gpt-oss-20b on TRT-LLM → **既存 vLLM Nemotron 3 Nano 30B-A3B-NVFP4** (port 8001) に相乗り
- TTS: Kokoro 82M の `lang_code` を `"a"` → `"j"`
- 画像生成: FLUX.1-Kontext-dev（非商用）→ **SDXL Turbo** に差し替え（CM Forum 2026 商用性懸念回避）
- エージェント: NeMo Agent Toolkit のまま、`agent-service` 配下の workflow を日本語プロンプトに

## License

Scripts: MIT License
