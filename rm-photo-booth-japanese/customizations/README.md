# Photo Booth Customizations

Reachy Mini Photo Booth Playbook v1.0.1 (commit `58d24e5`) に対して、
日本語化フェーズ着手時に試した改変パッチと、その過程で得た学びを
記録します。

## ファイル

| File | Purpose |
|---|---|
| `llm-service-nemotron-patch.yaml` | `docker-compose.yaml` の `llm` service を vLLM Nemotron Nano 9B v2 FP8 に差し替えるパッチ |
| `photobooth-llms-nemotron-patch.yml` | `agent-service/src/configs/photobooth.yml` の `llms` ブロックを Nemotron 用に差し替えるパッチ |
| `hardware_config-lowered-pid.yaml` | `robot-controller-service/data/hardware_config.yaml` の PID gain を Phase 1 default 相当に下げた版 (Input Voltage Error 仮説検証用、最終的に原因ではなかった) |

## Phase 2 の経緯

Phase 1（接続検証 + 19 サービス起動 + 写真撮影成功）が完了したのちに、
日本語化フェーズの最初として LLM を `openai/gpt-oss-20b` (TRT-LLM) から
`nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8` (vLLM) に差し替える検証を行いました。

結果として **vLLM Nemotron 起動は成功**（healthy、日本語応答も生成）した
ものの、その後の Photo Booth restart で Reachy が反応しなくなる現象が
発生し、最終的には `gpt-oss-20b` にロールバックしました。

その過程で見えた知見をこの `customizations/` に残しておきます。

## ハマりポイント 4 件 (vLLM Nemotron 起動)

1. **`HF_HUB_ENABLE_HF_TRANSFER=1` で起動失敗**
   - `pydantic ValidationError: 'hf_transfer' package is not available`
   - `nvcr.io/nvidia/vllm:25.12.post1-py3` image に `hf_transfer` パッケージが
     同梱されていないため。環境変数を削除すれば通常の HF download にフォールバック

2. **モデル id の正式表記は大文字スタイル**
   - 正: `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8`
   - VSS 3.1 EA の NIM 内部表記は小文字版だが、HF Hub には大文字版しかない
   - HF API で確認: `gated: false` / `license: nvidia-open-model-license`

3. **`--mamba_ssm_cache_dtype float32` 必須**
   - Nemotron H は Mamba-Transformer Hybrid アーキ
   - Mamba 層の SSM cache の dtype を float32 で明示しないと起動失敗
   - VSS 3.1 EA の停止 container `nvidia-nemotron-nano-9b-v2-fp8-shared-gpu` を
     `docker inspect` して発見

4. **tool-parser-plugin は VSS 3.1 EA 専用**
   - `--enable-auto-tool-choice` + `--tool-parser-plugin /opt/toolcall_parser/...`
     + `--tool-call-parser nemotron_json` は VSS 3.1 EA の独自実装で、
     external plugin file が必要
   - Photo Booth の NAT `photo_booth_react` はプレーンテキスト ReAct なので不要

## Reachy 不動化騒動 (Input Voltage Error)

LLM 差し替え後の Photo Booth restart で、`robot-controller` ログに
全 9 サーボから `Input Voltage Error` が連発する現象が発生しました。

**仮説 3 つすべて外れた**:

1. GPU 電源スパイク説 → NIM healthy 化後 GPU idle (11W) でも Error 継続で否定
2. PID gain 過大説 → `hardware_config.yaml` を default 相当に下げても Error 継続で否定
3. ハードウェア故障説 → USB-C ポート差し替え後の **Reachy 自動デモ動作目撃** で否定

**真相**: `Input Voltage Error` は **Dynamixel の警告レベル Hardware Error フラグ**
で、過去のラッチ履歴や瞬間電圧スパイクを daemon が polling 結果として吐いて
いるだけ。**実動作には影響しない**。

**最大の認知バイアス**: 「ログに ERROR と書いてあるから致命的」と思い込んだ。
**ログ ≠ 真実、実機 = 真実**。

## 真因: Photo Booth state machine が THINK で stuck

`interaction-manager` ログ精査で、agent が `ask_human` を complete した
あとに **STT が `disabled`、tracker が `OFF` に遷移してデッドロック**して
いたことが判明しました。

**復旧策**:

```bash
docker compose -p reachy-photo-booth restart \
  interaction-manager speech-to-text tracker
```

3 service を restart して state machine をリセットすると、自動で
`greet_user` が再 trigger され、Photo Booth サイクルが復活します。

## TTS 音量問題

写真生成・顔追従は動いたが、TTS の音声が Reachy スピーカーから聞こえ
ない問題がありました。`text-to-speech` container は wav を完全生成・
Kafka 送信していたので、原因は ALSA volume にありました。

**解決** (sudo 不要):

```bash
amixer -c 1 cset numid=5 60,60   # PCM Playback Volume (index=0) max
amixer -c 1 cset numid=6 60      # PCM Playback Volume (index=1) max
```

公式 `~/works/reachy-photo-booth/robot-controller-service/scripts/speaker_setup.sh`
は sudo + alsactl store のため、Claude Code の auto mode では deny された
ので、amixer 単独で代替しました。

## 適用手順

これらのパッチは「現時点ではロールバック済み・適用していない」状態の
記録です。再度 Nemotron 切り替えを試す場合は:

```bash
# 1. 編集 (patch ファイルの内容を該当ファイルに反映)
#    docker-compose.yaml の llm service ブロック
#    agent-service/src/configs/photobooth.yml の llms ブロック
#    robot-controller-service/data/hardware_config.yaml の PID 値

# 2. rebuild + restart
NVIDIA_API_KEY="$NGC_API_KEY" docker compose up -d --build \
  llm agent robot-controller
```

## ロールバック手順 (gpt-oss-20b に戻す)

backup ファイル (`*.orig-2026-05-15`) を復元して image を rebuild:

```bash
cd ~/works/reachy-photo-booth
cp docker-compose.yaml.orig-2026-05-15 docker-compose.yaml
cp agent-service/src/configs/photobooth.yml.orig-2026-05-15 \
   agent-service/src/configs/photobooth.yml
cp robot-controller-service/data/hardware_config.yaml.orig-2026-05-15 \
   robot-controller-service/data/hardware_config.yaml
NVIDIA_API_KEY="$NGC_API_KEY" docker compose up -d --build \
  llm agent robot-controller
```

## 次のフェーズ候補

- **日本語化方針再検討**: `gpt-oss-20b` は OpenAI 多言語対応で日本語応答可能。
  Nemotron 切り替えは必須ではない可能性
- **Kokoro lang_code `"a"` → `"j"`**: `text-to-speech-service` の 2 ファイル変更
- **STT 差し替え**: Parakeet 1.1B en-US → ReazonSpeech-NeMo-v2
- **UI i18n**: react-i18next で日本語化
- **画像生成差し替え**: FLUX.1-Kontext-dev (非商用) → SDXL Turbo (商用性懸念回避)
