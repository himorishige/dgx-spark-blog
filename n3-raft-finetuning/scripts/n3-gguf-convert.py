#!/usr/bin/env python3
"""N3 GGUF Conversion & Ollama Registration

Download LoRA adapter from SageMaker S3 output, convert to GGUF,
and register in Ollama with the base Nemotron 9B GGUF model.

Prerequisites:
  1. SageMaker training job completed
  2. llama.cpp cloned (for convert_lora_to_gguf.py)
  3. Base Nemotron 9B GGUF already imported in Ollama (N1 article setup)

Usage:
  # Download adapter from S3 and convert
  python n3-gguf-convert.py download \
    --job-name n3-qlora-20260221-164259 \
    --output-dir ./data/n3/adapter

  # Convert LoRA adapter to GGUF
  python n3-gguf-convert.py convert \
    --adapter-dir ./data/n3/adapter \
    --llama-cpp-dir ~/llama.cpp \
    --output-file ./data/n3/adapter/nemotron-9b-raft-lora.gguf

  # Create Ollama Modelfile and register
  python n3-gguf-convert.py register \
    --adapter-gguf ./data/n3/adapter/nemotron-9b-raft-lora.gguf \
    --base-model nemotron-9b-jp \
    --model-name nemotron-9b-jp-raft

  # All-in-one
  python n3-gguf-convert.py all \
    --job-name n3-qlora-20260221-164259 \
    --llama-cpp-dir ~/llama.cpp \
    --base-model nemotron-9b-jp
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

try:
    import boto3
except ImportError:
    print("Error: boto3 not installed. Run: pip install boto3")
    sys.exit(1)


DEFAULT_REGION = "ap-northeast-1"
S3_BUCKET_PREFIX = "sagemaker"
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
DEFAULT_OLLAMA_MODEL_NAME = "nemotron-9b-jp-raft"


def cmd_download(args):
    """Download LoRA adapter from SageMaker S3 output."""
    region = args.region
    job_name = args.job_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sm = boto3.client("sagemaker", region_name=region)

    # Get S3 output path from training job
    resp = sm.describe_training_job(TrainingJobName=job_name)
    status = resp["TrainingJobStatus"]
    if status != "Completed":
        print(f"Error: Training job '{job_name}' status is '{status}', not 'Completed'")
        sys.exit(1)

    s3_output = resp["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"=== Download Adapter ===")
    print(f"  Job: {job_name}")
    print(f"  S3 artifact: {s3_output}")

    if resp.get("BillableTimeInSeconds"):
        billable_min = resp["BillableTimeInSeconds"] / 60
        print(f"  Billable time: {billable_min:.1f} min")

    # Download model.tar.gz
    tar_path = output_dir / "model.tar.gz"
    print(f"  Downloading to: {tar_path}")

    # Parse S3 URI
    s3_parts = s3_output.replace("s3://", "").split("/", 1)
    bucket = s3_parts[0]
    key = s3_parts[1]

    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, key, str(tar_path))

    # Extract
    print(f"  Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(output_dir))

    # List extracted files
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        print(f"  Adapter directory: {adapter_dir}")
        for f in sorted(adapter_dir.iterdir()):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"  Warning: Expected 'adapter' subdirectory not found")
        print(f"  Extracted contents:")
        for f in sorted(output_dir.rglob("*")):
            if f.is_file() and f.name != "model.tar.gz":
                print(f"    {f.relative_to(output_dir)}")

    # Check for training metrics
    metrics_path = output_dir / "training_metrics.json"
    if metrics_path.exists():
        print(f"  Training metrics: {metrics_path}")

    print(f"\n  Download complete.")
    return str(adapter_dir)


def cmd_convert(args):
    """Convert LoRA adapter to GGUF format."""
    adapter_dir = Path(args.adapter_dir)
    llama_cpp_dir = Path(args.llama_cpp_dir)
    output_file = Path(args.output_file) if args.output_file else adapter_dir / "lora-adapter.gguf"
    outtype = args.outtype

    converter = llama_cpp_dir / "convert_lora_to_gguf.py"
    if not converter.exists():
        print(f"Error: {converter} not found")
        print(f"  Make sure llama.cpp is cloned at: {llama_cpp_dir}")
        sys.exit(1)

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"Error: adapter_config.json not found in {adapter_dir}")
        sys.exit(1)

    print(f"=== Convert LoRA to GGUF ===")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Output: {output_file}")
    print(f"  Output type: {outtype}")
    print(f"  Base model ID: {BASE_MODEL_ID}")

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
        print(f"\nError: Output file not created")
        sys.exit(1)

    return str(output_file)


def cmd_register(args):
    """Create Ollama Modelfile and register the RAFT model."""
    adapter_gguf = Path(args.adapter_gguf).resolve()
    base_model = args.base_model
    model_name = args.model_name

    if not adapter_gguf.exists():
        print(f"Error: {adapter_gguf} not found")
        sys.exit(1)

    print(f"=== Register in Ollama ===")
    print(f"  Base model: {base_model}")
    print(f"  Adapter GGUF: {adapter_gguf}")
    print(f"  New model name: {model_name}")

    # Verify base model exists in Ollama
    result = subprocess.run(
        ["ollama", "show", base_model],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: Base model '{base_model}' not found in Ollama")
        print(f"  Run 'ollama list' to see available models")
        sys.exit(1)

    # Create Modelfile
    modelfile_path = adapter_gguf.parent / "Modelfile"
    modelfile_content = f"""FROM {base_model}
ADAPTER {adapter_gguf}
"""
    modelfile_path.write_text(modelfile_content)
    print(f"  Modelfile: {modelfile_path}")

    # Register
    print(f"  Creating Ollama model...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=False,
    )

    if result.returncode != 0:
        print(f"\nError: Ollama registration failed")
        sys.exit(1)

    print(f"\n  Registered: {model_name}")
    print(f"  Test with: ollama run {model_name} 'テスト'")


def cmd_all(args):
    """Download, convert, and register in one step."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    args.output_dir = str(output_dir)
    adapter_dir = cmd_download(args)

    # Step 2: Convert
    args.adapter_dir = adapter_dir
    args.output_file = str(output_dir / "nemotron-9b-raft-lora.gguf")
    args.outtype = args.outtype if hasattr(args, "outtype") and args.outtype else "f16"
    adapter_gguf = cmd_convert(args)

    # Step 3: Register
    args.adapter_gguf = adapter_gguf
    args.model_name = args.model_name if hasattr(args, "model_name") and args.model_name else DEFAULT_OLLAMA_MODEL_NAME
    cmd_register(args)


def main():
    parser = argparse.ArgumentParser(description="N3 GGUF Conversion & Ollama Registration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    dl = subparsers.add_parser("download", help="Download adapter from S3")
    dl.add_argument("--job-name", required=True, help="SageMaker training job name")
    dl.add_argument("--output-dir", default="./data/n3/adapter", help="Output directory")
    dl.add_argument("--region", default=DEFAULT_REGION)
    dl.set_defaults(func=cmd_download)

    # convert
    cv = subparsers.add_parser("convert", help="Convert LoRA adapter to GGUF")
    cv.add_argument("--adapter-dir", required=True, help="PEFT adapter directory")
    cv.add_argument("--llama-cpp-dir", required=True, help="llama.cpp directory")
    cv.add_argument("--output-file", help="Output GGUF file path")
    cv.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    cv.set_defaults(func=cmd_convert)

    # register
    rg = subparsers.add_parser("register", help="Register in Ollama")
    rg.add_argument("--adapter-gguf", required=True, help="GGUF adapter file")
    rg.add_argument("--base-model", default="nemotron-9b-jp", help="Ollama base model name")
    rg.add_argument("--model-name", default=DEFAULT_OLLAMA_MODEL_NAME, help="New Ollama model name")
    rg.set_defaults(func=cmd_register)

    # all
    al = subparsers.add_parser("all", help="Download, convert, and register")
    al.add_argument("--job-name", required=True, help="SageMaker training job name")
    al.add_argument("--output-dir", default="./data/n3/adapter", help="Output directory")
    al.add_argument("--region", default=DEFAULT_REGION)
    al.add_argument("--llama-cpp-dir", required=True, help="llama.cpp directory")
    al.add_argument("--base-model", default="nemotron-9b-jp", help="Ollama base model name")
    al.add_argument("--model-name", default=DEFAULT_OLLAMA_MODEL_NAME, help="New Ollama model name")
    al.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    al.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
