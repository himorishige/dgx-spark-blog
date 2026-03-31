"""SAM3 vs SAM3.1 video tracking benchmark.

Run inside the SAM3 container:
    python /bench/sam3-video-bench.py /data/video.mp4 --prompts "person,hard hat"
"""

import argparse
import os
import time

import numpy as np
import torch

# --- DGX Spark dtype compatibility patches ---
import torch.nn.functional as F

_orig_linear = F.linear
def _safe_linear(input, weight, bias=None):
    return _orig_linear(input.to(weight.dtype), weight, bias)
F.linear = _safe_linear

_orig_sdpa = F.scaled_dot_product_attention
def _safe_sdpa(q, k, v, *a, **kw):
    return _orig_sdpa(q, k, v, *a, **kw).to(q.dtype)
F.scaled_dot_product_attention = _safe_sdpa
# --- End patches ---

from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor


def extract_frames(video_path: str, max_frames: int = 30, fps: int = 5) -> list:
    """Extract frames from video using ffmpeg."""
    import subprocess
    import tempfile
    from PIL import Image

    tmpdir = tempfile.mkdtemp()
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        f"{tmpdir}/frame_%04d.jpg",
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)

    frames = []
    for f in sorted(os.listdir(tmpdir)):
        if f.endswith(".jpg"):
            img = Image.open(os.path.join(tmpdir, f)).convert("RGB")
            frames.append(np.array(img))
    return frames


def bench_image_sequential(frames: list, prompts: list[str], threshold: float = 0.5):
    """Benchmark: process each frame independently with image model (SAM3 style)."""
    print("\n=== SAM3 Image Model (sequential, per-frame) ===")
    model = build_sam3_image_model()
    from sam3.model.sam3_image_processor import Sam3Processor
    processor = Sam3Processor(model)

    total_detections = 0
    t0 = time.time()

    for i, frame_np in enumerate(frames):
        from PIL import Image
        img = Image.fromarray(frame_np)
        img = img.resize((1024, int(1024 * img.height / img.width)))
        state = processor.set_image(img)
        for prompt in prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt.strip())
            scores = output["scores"]
            keep = scores >= threshold
            total_detections += keep.sum().item()

    elapsed = time.time() - t0
    fps = len(frames) / elapsed
    print(f"  Frames: {len(frames)}")
    print(f"  Prompts: {prompts}")
    print(f"  Total detections: {total_detections}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Per-frame: {elapsed/len(frames)*1000:.0f}ms")
    return {"mode": "image_sequential", "frames": len(frames), "elapsed_s": round(elapsed, 1), "fps": round(fps, 1), "detections": total_detections}


def bench_video_predictor(frames: list, prompts: list[str], threshold: float = 0.5):
    """Benchmark: use SAM3 video predictor with object multiplexing."""
    print("\n=== SAM3.1 Video Predictor (multiplex) ===")

    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        predictor = build_sam3_multiplex_video_predictor()
        mode_name = "multiplex_video"
        print("  Using: build_sam3_multiplex_video_predictor (SAM 3.1)")
    except (ImportError, Exception) as e:
        print(f"  Multiplex not available ({e}), falling back to standard video predictor")
        predictor = build_sam3_video_predictor()
        mode_name = "standard_video"
        print("  Using: build_sam3_video_predictor (SAM 3)")

    t0 = time.time()

    # Initialize with video frames
    with torch.inference_mode():
        state = predictor.init_state(frames)

        # Add prompts as text conditions on first frame
        for prompt_idx, prompt in enumerate(prompts):
            predictor.add_new_text_prompt(
                state=state,
                frame_idx=0,
                obj_id=prompt_idx + 1,
                text=prompt.strip(),
            )

        # Propagate through all frames
        total_detections = 0
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask in zip(obj_ids, masks):
                if mask is not None:
                    binary = (mask.squeeze().cpu().numpy() > 0.5)
                    if binary.any():
                        total_detections += 1

    elapsed = time.time() - t0
    fps = len(frames) / elapsed
    print(f"  Frames: {len(frames)}")
    print(f"  Prompts: {prompts}")
    print(f"  Total detections: {total_detections}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  Per-frame: {elapsed/len(frames)*1000:.0f}ms")
    return {"mode": mode_name, "frames": len(frames), "elapsed_s": round(elapsed, 1), "fps": round(fps, 1), "detections": total_detections}


def main():
    parser = argparse.ArgumentParser(description="SAM3 vs SAM3.1 video tracking benchmark")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--prompts", default="person,hard hat,safety vest", help="Comma-separated prompts")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to process")
    parser.add_argument("--fps", type=int, default=5, help="Frame extraction FPS")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",")]
    print(f"Video: {args.video}")
    print(f"Prompts: {prompts}")
    print(f"Max frames: {args.max_frames}, FPS: {args.fps}")

    frames = extract_frames(args.video, args.max_frames, args.fps)
    print(f"Extracted {len(frames)} frames ({frames[0].shape})")

    # Benchmark 1: Image model (SAM3 style)
    result_img = bench_image_sequential(frames, prompts, args.threshold)

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Benchmark 2: Video predictor (SAM3.1 multiplex if available)
    result_vid = bench_video_predictor(frames, prompts, args.threshold)

    # Summary
    print("\n=== Summary ===")
    print(f"  Image sequential: {result_img['fps']:.1f} FPS ({result_img['elapsed_s']}s)")
    print(f"  Video predictor:  {result_vid['fps']:.1f} FPS ({result_vid['elapsed_s']}s)")
    if result_img['fps'] > 0:
        speedup = result_vid['fps'] / result_img['fps']
        print(f"  Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
