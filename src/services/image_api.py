"""Paid image generation through OpenRouter's Images API, for the beats without good stock
(the other generator is ComfyUI, local and free: services/comfy.py).

    POST https://openrouter.ai/api/v1/images  {"model", "prompt", "aspect_ratio", "resolution"}
    -> {"data": [{"b64_json", "media_type"}], "usage": {"cost"}}

    path, cost = generate("Crowd marching in a Madrid street...", "out", 1080, 1920,
                          "google/gemini-nano-banana-2.1", "2K")

Every model takes aspect_ratio (16:9, 9:16...); resolution only some (see each model in
GET /api/v1/images/models). The key is OPENROUTER_API_KEY in the .env (it needs credit).
"""
import base64
import os
from pathlib import Path

import requests

from llmkit import config  # noqa: F401  (loads the .env)

URL = "https://openrouter.ai/api/v1/images"
TIMEOUT = 180
RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"]
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def aspect_ratio(width, height):
    """The common ratio closest to the frame's."""
    def value(r):
        a, b = r.split(":")
        return int(a) / int(b)
    return min(RATIOS, key=lambda r: abs(value(r) - width / height))


def generate(prompt, dst, width, height, model, resolution=None):
    """prompt (English) -> the image saved as dst + its extension. Returns (path, cost in $)."""
    payload = {"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio(width, height)}
    if resolution:
        payload["resolution"] = resolution
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in the .env file")
    r = requests.post(URL, json=payload, timeout=TIMEOUT, headers={"Authorization": f"Bearer {key}"})
    if not r.ok:
        raise RuntimeError(f"OpenRouter images {r.status_code}: {r.text[:500]}")
    data = r.json()
    image = (data.get("data") or [{}])[0]
    if not image.get("b64_json"):
        raise RuntimeError(f"OpenRouter images: no image in the answer: {str(data)[:500]}")
    path = Path(dst).with_suffix(EXTENSIONS.get(image.get("media_type"), ".png"))
    path.write_bytes(base64.b64decode(image["b64_json"]))
    return path, float((data.get("usage") or {}).get("cost") or 0)
