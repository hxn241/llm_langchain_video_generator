"""ComfyUI over HTTP: image generation (Z-Image Turbo) for scenes without good stock.

ComfyUI runs on its own (its own folder and venv, see README) and this talks to its API:
    POST /prompt (a workflow in API format) -> poll /history/<id> -> GET /view (the image)
    POST /free   unloads its models (VRAM for the next phase)

    path = generate("Crowd marching in a Madrid street...", "out.png", width=1440, height=2560)

config.yaml, section `comfy`: url, model, text_encoder, vae, steps, megapixels...
The models go in ComfyUI/models/ (diffusion_models/, text_encoders/, vae/), from
huggingface.co/Comfy-Org/z_image_turbo (split_files/).
"""
import json
import math
import random
import time
import uuid
from pathlib import Path

import requests

from services.settings import load_settings

# As in ComfyUI's template (image_z_image_turbo): a distilled model, few steps and no CFG
Z_IMAGE = {"model": "z_image_turbo_bf16.safetensors", "text_encoder": "qwen_3_4b.safetensors",
           "vae": "ae.safetensors", "steps": 8, "cfg": 1.0, "shift": 3.0,
           "sampler": "res_multistep", "scheduler": "simple"}
SIZE_STEP = 32            # width and height are multiples of this


def options(settings=None):
    return (settings or load_settings()).get("comfy", {})


def url(settings=None):
    return options(settings).get("url", "http://127.0.0.1:8188").rstrip("/")


def running(settings=None):
    try:
        return requests.get(f"{url(settings)}/system_stats", timeout=3).ok
    except requests.RequestException:
        return False


def generation_size(width, height, megapixels):
    """The frame's aspect ratio at about `megapixels` (the image is then scaled to the frame)."""
    scale = math.sqrt(megapixels * 1e6 / (width * height))
    return (max(SIZE_STEP, round(width * scale / SIZE_STEP) * SIZE_STEP),
            max(SIZE_STEP, round(height * scale / SIZE_STEP) * SIZE_STEP))


def z_image_workflow(prompt, width, height, seed, o):
    """Text-to-image with Z-Image Turbo, in ComfyUI's API format (node id -> class + inputs).
    `o`: Z_IMAGE with config.yaml's comfy section on top."""
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": o["model"], "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": o["shift"]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": o["text_encoder"], "type": "lumina2",
                                                        "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": o["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},   # no CFG: no negative
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "seed": seed, "steps": o["steps"], "cfg": o["cfg"], "sampler_name": o["sampler"],
            "scheduler": o["scheduler"], "positive": ["pos", 0], "negative": ["neg", 0],
            "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "llm_video"}},
    }


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
    """prompt (English) -> an image with the frame's aspect ratio (width:height), saved to dst."""
    o = Z_IMAGE | options(settings)
    w, h = generation_size(width, height, o.get("megapixels", 2.0))
    workflow = z_image_workflow(prompt, w, h, seed if seed is not None else random.randrange(2**32), o)
    Path(dst).write_bytes(run(workflow, o.get("timeout", 900), settings))
    return Path(dst)


def free(settings=None):
    """Unloads ComfyUI's models from the VRAM (before the LLM or the TTS need it)."""
    try:
        requests.post(f"{url(settings)}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
    except requests.RequestException:
        pass
