#!/usr/bin/env python3
"""N5 GGUF Conversion & Ollama Registration

Convert SimPO/DPO LoRA adapter to GGUF and register in Ollama.
Adapted from n3-gguf-convert.py (simplified: no S3 download, local only).

Prerequisites:
  1. Training completed (adapter in data/n5/adapter/)
  2. llama.cpp cloned (for convert_lora_to_gguf.py)
  3. Base Nemotron 9B GGUF already imported in Ollama (N1 article setup)

Usage:
  # Convert LoRA adapter to GGUF
  python n5-gguf-convert.py convert \
    --adapter-dir ./data/n5/adapter \
    --llama-cpp-dir ~/llama.cpp

  # Register in Ollama
  python n5-gguf-convert.py register \
    --adapter-gguf ./data/n5/adapter/nemotron-9b-cai-lora.gguf

  # All-in-one
  python n5-gguf-convert.py all \
    --adapter-dir ./data/n5/adapter \
    --llama-cpp-dir ~/llama.cpp
"""

import argparse
import subprocess
import sys
from pathlib import Path


BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
DEFAULT_BASE_OLLAMA = "nemotron-9b-jp-nothink"
DEFAULT_MODEL_NAME = "nemotron-9b-jp-cai"
DEFAULT_GGUF_NAME = "nemotron-9b-cai-lora.gguf"


def cmd_convert(args):
    """Convert LoRA adapter to GGUF format."""
    adapter_dir = Path(args.adapter_dir)
    llama_cpp_dir = Path(args.llama_cpp_dir)
    output_file = Path(args.output_file) if args.output_file else adapter_dir / DEFAULT_GGUF_NAME
    outtype = args.outtype

    converter = llama_cpp_dir / "convert_lora_to_gguf.py"
    if not converter.exists():
        print(f"Error: {converter} not found")
        sys.exit(1)

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"Error: adapter_config.json not found in {adapter_dir}")
        sys.exit(1)

    print("=== Convert LoRA to GGUF ===")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Output: {output_file}")
    print(f"  Output type: {outtype}")

    cmd = [
        sys.executable, str(converter),
        str(adapter_dir),
        "--base-model-id", BASE_MODEL_ID,
        "--outtype", outtype,
        "--outfile", str(output_file),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nError: Conversion failed (exit code {result.returncode})")
        sys.exit(1)

    if output_file.exists():
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"\n  GGUF adapter created: {output_file} ({size_mb:.1f} MB)")
    else:
        print("\nError: Output file not created")
        sys.exit(1)

    return str(output_file)


def cmd_register(args):
    """Create Ollama Modelfile and register."""
    adapter_gguf = Path(args.adapter_gguf).resolve()
    base_model = args.base_model
    model_name = args.model_name

    if not adapter_gguf.exists():
        print(f"Error: {adapter_gguf} not found")
        sys.exit(1)

    print("=== Register in Ollama ===")
    print(f"  Base model: {base_model}")
    print(f"  Adapter: {adapter_gguf}")
    print(f"  New model: {model_name}")

    # Verify base model
    result = subprocess.run(
        ["ollama", "show", base_model],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: Base model '{base_model}' not found in Ollama")
        sys.exit(1)

    # Create Modelfile
    modelfile_path = adapter_gguf.parent / "Modelfile"
    modelfile_content = f"FROM {base_model}\nADAPTER {adapter_gguf}\n"
    modelfile_path.write_text(modelfile_content)

    # Register
    print("  Creating Ollama model...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=False,
    )
    if result.returncode != 0:
        print("\nError: Ollama registration failed")
        sys.exit(1)

    print(f"\n  Registered: {model_name}")
    print(f"  Test: ollama run {model_name} 'テスト'")


def cmd_all(args):
    """Convert and register in one step."""
    args.output_file = None
    adapter_gguf = cmd_convert(args)
    args.adapter_gguf = adapter_gguf
    cmd_register(args)


def main():
    parser = argparse.ArgumentParser(description="N5 GGUF Conversion & Ollama Registration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # convert
    cv = subparsers.add_parser("convert", help="Convert LoRA to GGUF")
    cv.add_argument("--adapter-dir", required=True)
    cv.add_argument("--llama-cpp-dir", required=True)
    cv.add_argument("--output-file", help="Output GGUF path")
    cv.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    cv.set_defaults(func=cmd_convert)

    # register
    rg = subparsers.add_parser("register", help="Register in Ollama")
    rg.add_argument("--adapter-gguf", required=True)
    rg.add_argument("--base-model", default=DEFAULT_BASE_OLLAMA)
    rg.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    rg.set_defaults(func=cmd_register)

    # all
    al = subparsers.add_parser("all", help="Convert and register")
    al.add_argument("--adapter-dir", required=True)
    al.add_argument("--llama-cpp-dir", required=True)
    al.add_argument("--base-model", default=DEFAULT_BASE_OLLAMA)
    al.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    al.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    al.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
