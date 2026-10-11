"""2.5D worker: photos -> video segments where the foreground moves a little more than the
background (parallax). Runs with .venv-tts (torch + transformers), NOT with the project's venv:

    .venv-tts/Scripts/python.exe src/tts_worker/parallax_worker.py job.json

job.json (written by graph/nodes/assemble.py):
    {"model": "depth-anything/Depth-Anything-V2-Small-hf", "strength": 0.02, "zoom": 0.1,
     "ffmpeg": "...ffmpeg.exe", "width": 1920, "height": 1080, "fps": 30, "encode": ["-c:v", ...],
     "items": [{"image": "05.jpg", "out": "shot_005.mp4", "frames": 180,
                "motion": {"z0": 1.0, "z1": 1.1, "x0": 0, "y0": 0, "x1": 0.7, "y1": -0.3}}, ...]}

Prints one JSON line at the end:
    {"ok": true, "items": [{"out": "...", "ok": true, "flat": false}, ...], "seconds": ...}

How: the depth model gives each pixel of the photo how near it is (0 far .. 1 near). The camera
moves as in assemble.camera (same maths), and each output pixel is then shifted by its depth:
along the pan for pans (near things slide more), towards/away from the centre for zooms (near
things grow faster). The depth is widened around the foreground and blurred, so the edges bend
softly instead of tearing. A flat photo (chart, screenshot) gets the plain move.
"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from transformers import pipeline  # noqa: E402

FLAT = 0.15                      # depth spread (relative) below this: a flat image, no parallax
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def camera(motion, big_w, big_h, t):
    """= graph/nodes/assemble.camera: (centre x, centre y, w, h) of the frame in the enlarged photo."""
    e = t * t * (3 - 2 * t)
    zoom = motion["z0"] + (motion["z1"] - motion["z0"]) * e
    w, h = big_w / zoom, big_h / zoom
    x = motion["x0"] + (motion["x1"] - motion["x0"]) * e
    y = motion["y0"] + (motion["y1"] - motion["y0"]) * e
    return big_w / 2 + x * (big_w - w) / 2, big_h / 2 + y * (big_h - h) / 2, w, h, e


def depth_map(estimator, image):
    """(1, 1, H, W) depth of the photo, 0 far .. 1 near, widened around the foreground and blurred;
    and whether the photo is flat."""
    raw = estimator(image)["predicted_depth"].squeeze().float().to(DEVICE)      # bigger = nearer
    lo, hi = torch.quantile(raw.flatten(), torch.tensor([0.02, 0.98], device=DEVICE))
    flat = bool((hi - lo) / hi.abs().clamp(min=1e-6) < FLAT)
    d = ((raw - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)[None, None]
    d = F.interpolate(d, size=(image.height, image.width), mode="bilinear", align_corners=False)
    k = max(3, image.width // 120) | 1                                            # odd, ~1 % of the width
    d = F.max_pool2d(d, k, 1, k // 2)                                             # the foreground a bit wider
    for _ in range(2):
        d = F.avg_pool2d(d, k, 1, k // 2, count_include_pad=False)                # soft edges
    return d, flat


def sample(tensor, x, y, width, height, mode="bilinear"):
    """tensor (1, C, h, w) at the points x, y (pixels of the enlarged photo, 0 = left/top edge)."""
    grid = torch.stack((x / width * 2 - 1, y / height * 2 - 1), dim=-1)[None]
    return F.grid_sample(tensor, grid, mode=mode, padding_mode="border", align_corners=False)


def render(job, item, estimator):
    width, height, zoom = job["width"], job["height"], job["zoom"]
    image = ImageOps.fit(Image.open(item["image"]).convert("RGB"),
                         (round(width * (1 + zoom)), round(height * (1 + zoom))), Image.Resampling.LANCZOS)
    big_w, big_h = image.size
    depth, flat = depth_map(estimator, image)
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).to(DEVICE)
    pixels = pixels.view(big_h, big_w, 3).permute(2, 0, 1)[None].float() / 255

    m = item["motion"]
    pan = torch.tensor([m["x1"] - m["x0"], m["y1"] - m["y0"]], dtype=torch.float32)
    pan = pan / pan.norm() if pan.norm() > 0 else pan                              # the pan's direction
    zoom_way = (m["z1"] > m["z0"]) - (m["z1"] < m["z0"])                           # 1 in, -1 out, 0 none
    strength = 0 if flat else job["strength"]
    u = (torch.arange(width, device=DEVICE) + 0.5) / width - 0.5
    v = (torch.arange(height, device=DEVICE) + 0.5) / height - 0.5
    v, u = torch.meshgrid(v, u, indexing="ij")

    process = subprocess.Popen([job["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                "-r", str(job["fps"]), "-i", "-", "-frames:v", str(item["frames"]),
                                *job["encode"], item["out"]], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with torch.no_grad():
            for n in range(item["frames"]):
                cx, cy, w, h, e = camera(m, big_w, big_h, n / max(1, item["frames"] - 1))
                p = 2 * e - 1                                    # -1 .. 1 along the move
                lateral_x, lateral_y = (strength * big_w * p * pan).tolist()
                radial = 2 * strength * p * zoom_way
                x0, y0 = cx + u * w, cy + v * h
                x, y = x0, y0
                for _ in range(2):                               # the depth where the pixel really comes from
                    dd = sample(depth, x, y, big_w, big_h)[0, 0] - 0.5
                    scale = 1 + radial * dd
                    x = cx + (x0 - cx) / scale + lateral_x * dd
                    y = cy + (y0 - cy) / scale + lateral_y * dd
                frame = sample(pixels, x, y, big_w, big_h, mode="bicubic")[0].clamp(0, 1)
                process.stdin.write((frame * 255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    error = process.stderr.read().decode("utf-8", "replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {error[-500:]}")
    return flat


def main(job_path):
    job = json.loads(open(job_path, encoding="utf-8-sig").read())
    started = time.time()
    estimator = pipeline("depth-estimation", model=job["model"], device=DEVICE)
    items = []
    for item in job["items"]:
        try:
            flat = render(job, item, estimator)
            items.append({"out": item["out"], "ok": True, "flat": flat})
        except Exception as e:                                    # noqa: BLE001 - that photo: plain move
            print(f"{item['image']}: {e}", file=sys.stderr)
            items.append({"out": item["out"], "ok": False})
    print(json.dumps({"ok": True, "items": items, "seconds": round(time.time() - started, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
