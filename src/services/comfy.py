"""ComfyUI over HTTP: image generation (HiDream-O1-Image) for scenes without good stock.

ComfyUI runs on its own (its own folder and venv, see README) and this talks to its API:
    POST /prompt (a workflow in API format) -> poll /history/<id> -> GET /view (the image)
    POST /free   unloads its models (VRAM for the next phase)

    path = generate("Crowd marching in a Madrid street...", "out.png", width=1440, height=2560)

config.yaml, section `comfy`: url, checkpoint, variant (dev | full), steps...
"""
import json
import random
import time
import uuid
from pathlib import Path

import requests

from services.settings import load_settings

# Resolutions HiDream-O1 was trained on (ComfyUI's template): one per aspect ratio
TRAINED_SIZES = [(2048, 2048), (2304, 1728), (2304, 1792), (2496, 1664), (2560, 1440), (3104, 1312),
                 (1728, 2304), (1792, 2304), (1664, 2496), (1440, 2560), (1312, 3104)]

# The two variants, as in ComfyUI's templates (image_hidream_o1_dev / image_hidream_o1)
VARIANTS = {
    "dev":  {"steps": 28, "cfg": 1.0, "noise_scale": 7.6, "sampler": "lcm"},
    "full": {"steps": 40, "cfg": 5.0, "noise_scale": 8.0, "sampler": "dpmpp_2m_sde_gpu", "seam_smoothing": True},
}


def options(settings=None):
    return (settings or load_settings()).get("comfy", {})


def url(settings=None):
    return options(settings).get("url", "http://127.0.0.1:8188").rstrip("/")


def running(settings=None):
    try:
        return requests.get(f"{url(settings)}/system_stats", timeout=3).ok
    except requests.RequestException:
        return False


def trained_size(width, height):
    """The trained resolution closest to the frame's aspect ratio (the image is then scaled to the frame)."""
    ratio = width / height
    return min(TRAINED_SIZES, key=lambda s: abs(s[0] / s[1] - ratio))


def hidream_workflow(prompt, width, height, seed, checkpoint, variant="dev", steps=None, negative=""):
    """Text-to-image with HiDream-O1-Image, in ComfyUI's API format (node id -> class + inputs)."""
    v = VARIANTS[variant]
    model = ["noise", 0]
    nodes = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "noise": {"class_type": "ModelNoiseScale", "inputs": {"model": ["ckpt", 0], "noise_scale": v["noise_scale"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": negative}},
        "latent": {"class_type": "EmptyHiDreamO1LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    if v.get("seam_smoothing"):
        nodes["seam"] = {"class_type": "HiDreamO1PatchSeamSmoothing",
                         "inputs": {"model": model, "start_percent": 0.8, "end_percent": 1.0, "pattern": "single_shift",
                                    "passes": "ramp_2_4", "blend": "median", "strength": 1.0}}
        model = ["seam", 0]
    if v["sampler"] == "lcm":
        nodes["sampler"] = {"class_type": "SamplerLCM", "inputs": {"s_noise": 1.0, "s_noise_end": 1.0, "noise_clip_std": 2.5}}
    else:
        nodes["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": v["sampler"]}}
    nodes |= {
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "normal",
                                                              "steps": steps or v["steps"], "denoise": 1.0}},
        "sample": {"class_type": "SamplerCustom", "inputs": {
            "model": model, "add_noise": True, "noise_seed": seed, "cfg": v["cfg"], "positive": ["pos", 0],
            "negative": ["neg", 0], "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "llm_video"}},
    }
    return nodes


def run(workflow, timeout=900, settings=None):
    """Queues a workflow and waits; returns the first image's bytes."""
    base = url(settings)
    r = requests.post(f"{base}/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex}, timeout=30)
    if not r.ok:
        raise RuntimeError(f"ComfyUI rejected the workflow ({r.status_code}): {r.text[:800]}")
    prompt_id = r.json()["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = requests.get(f"{base}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                messages = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI failed: {json.dumps(messages, ensure_ascii=False)[:1500]}")
            for output in history.get("outputs", {}).values():
                for image in output.get("images", []):
                    return requests.get(f"{base}/view", params=image, timeout=60).content
            if status.get("completed"):
                raise RuntimeError("ComfyUI finished without an image")
        time.sleep(1)
    raise TimeoutError(f"ComfyUI did not finish in {timeout} s")


def generate(prompt, dst, width, height, seed=None, settings=None):
    """prompt (English) -> an image at the trained size closest to width:height, saved to dst."""
    o = options(settings)
    w, h = trained_size(width, height)
    workflow = hidream_workflow(prompt, w, h, seed if seed is not None else random.randrange(2**32),
                                o.get("checkpoint", "hidream_o1_image_dev_fp8_scaled.safetensors"),
                                o.get("variant", "dev"), o.get("steps"))
    Path(dst).write_bytes(run(workflow, o.get("timeout", 900), settings))
    return Path(dst)


def free(settings=None):
    """Unloads ComfyUI's models from the VRAM (before the LLM or the TTS need it)."""
    try:
        requests.post(f"{url(settings)}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
    except requests.RequestException:
        pass
