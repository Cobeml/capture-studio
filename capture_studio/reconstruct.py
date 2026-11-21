"""Stage 1 -> 2: RGB phone video -> LingBot-Map -> coloured point cloud + camera path.

Writes data/scan/raw_predictions.npz (points, colours, confidence, frame index, camera poses).
Everything downstream (BIM fitting, viewer) reads that file, so this GPU step runs once.

    python -m capture_studio.reconstruct --video data/site.mp4 --fps 6
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
LINGBOT = ROOT / "third_party" / "lingbot-map"
sys.path.insert(0, str(LINGBOT))

import demo as lingbot_demo  # noqa: E402  (LingBot-Map's own loader / post-processing)
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "site.mp4"))
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--model", default=str(ROOT / "models" / "lingbot-map" / "lingbot-map.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "scan" / "raw_predictions.npz"))
    ap.add_argument("--keyframe_interval", type=int, default=1, help=">1 keeps GPU memory flat on long clips")
    ap.add_argument("--window", type=int, default=32, help="KV-cache sliding window (frames); lower = less GPU memory")
    ap.add_argument("--use_sdpa", action="store_true", help="skip FlashInfer, use native attention")
    a = ap.parse_args()

    try:
        import flashinfer  # noqa: F401
    except ImportError:
        a.use_sdpa = True       # FlashInfer is optional; native attention works everywhere
    device = torch.device("cuda")
    images, paths, _ = lingbot_demo.load_images(video_path=a.video, fps=a.fps)
    margs = SimpleNamespace(
        mode="streaming", image_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
        kv_cache_sliding_window=a.window, num_scale_frames=8, use_sdpa=a.use_sdpa,
        camera_num_iterations=4, model_path=a.model,
    )
    model = lingbot_demo.load_model(margs, device)
    dtype = torch.bfloat16
    model.aggregator = model.aggregator.to(dtype=dtype)

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model.inference_streaming(
            images.to(device), num_scale_frames=8, keyframe_interval=a.keyframe_interval,
            output_device=torch.device("cpu"),
        )
    secs = time.time() - t0
    print(f"LingBot-Map: {images.shape[0]} frames in {secs:.1f}s, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    pred, images_cpu = lingbot_demo.postprocess(pred, pred.get("images", images))
    images_cpu = images_cpu.reshape(-1, *images_cpu.shape[-3:])
    extr = pred["extrinsic"].float().numpy()            # (S,3,4) world->cam
    intr = pred["intrinsic"].float().numpy()            # (S,3,3)
    depth = pred["depth"].float().numpy()               # (S,H,W,1)
    conf = pred["depth_conf"].float().numpy()           # (S,H,W)
    pts = unproject_depth_map_to_point_map(depth, extr, intr)  # (S,H,W,3)
    rgb = (images_cpu.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, points=pts.astype(np.float32), rgb=rgb, conf=conf.astype(np.float32),
        extrinsic=extr, intrinsic=intr, fps=a.fps, inference_seconds=secs,
    )
    print("saved", a.out, pts.shape)


if __name__ == "__main__":
    main()
