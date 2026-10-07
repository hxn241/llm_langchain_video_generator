"""Stock footage and photos: Pexels and Pixabay (free, real footage with a licence to use it).
One of the sources of the visuals node (graph/nodes/visuals.py).

    search(query, kind, orientation, providers)   -> [candidate, ...] in the providers' order
    thumbnail(candidate, folder)                   -> local JPEG (to compare it with the text)
    download(candidate, width, height, folder)     -> local file, in the smallest size that fills the frame

A candidate is a plain dict, the same for every provider:
    {"provider", "id", "kind": "video" | "photo", "page_url", "author", "width", "height", "thumb",
     "duration" (videos, s), "ai_generated", "renditions": [{"url", "width", "height"}, ...]}

Keys in the .env: PEXELS_API_KEY, PIXABAY_API_KEY. A provider without a key is skipped.
Rate limits: Pixabay 100 requests/min, Pexels 200/hour.
"""
import os
import time
from pathlib import Path

import requests

from llmkit import config  # noqa: F401  (loads the .env)

TIMEOUT = 20
MAX_UPSCALE = 1.35        # a rendition may be enlarged this much to fill the frame (more looks blurry)
KEYS = {"pexels": "PEXELS_API_KEY", "pixabay": "PIXABAY_API_KEY"}
HEADERS = {"User-Agent": "Mozilla/5.0"}


def available(providers):
    """The providers (in order) that have a key in the .env."""
    return [p for p in providers if os.getenv(KEYS[p], "").strip()]


def _get(url, params=None, headers=None, retries=2):
    """GET with a wait-and-retry on 429 (rate limit)."""
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        if r.status_code == 429 and attempt < retries:
            time.sleep(int(r.headers.get("X-RateLimit-Reset") or r.headers.get("Retry-After") or 30) + 1)
            continue
        r.raise_for_status()
        return r.json()


def _fit(width, height, max_width=None, long_side=None):
    scale = min(1.0, max_width / width) if max_width else min(1.0, long_side / max(width, height))
    return {"width": round(width * scale), "height": round(height * scale)}


# ----------------------------------------------------------------- pexels
def _pexels(query, kind, orientation, per_page):
    headers = {"Authorization": os.environ[KEYS["pexels"]]}
    params = {"query": query, "orientation": orientation, "per_page": per_page}
    if kind == "video":
        data = _get("https://api.pexels.com/videos/search", params, headers)
        return [{"provider": "pexels", "id": v["id"], "kind": "video", "page_url": v["url"],
                 "author": (v.get("user") or {}).get("name", ""), "width": v["width"], "height": v["height"],
                 "duration": v["duration"], "ai_generated": False, "thumb": v.get("image", ""),
                 "renditions": [{"url": f["link"], "width": f["width"], "height": f["height"]}
                                for f in v["video_files"] if f.get("file_type") == "video/mp4"]}
                for v in data.get("videos", [])]
    data = _get("https://api.pexels.com/v1/search", params, headers)
    return [{"provider": "pexels", "id": p["id"], "kind": "photo", "page_url": p["url"],
             "author": p.get("photographer", ""), "width": p["width"], "height": p["height"],
             "ai_generated": False, "thumb": p["src"]["medium"],
             # the original, resized by the CDN to at most 2400 px wide (aspect kept)
             "renditions": [{"url": p["src"]["original"] + "?auto=compress&cs=tinysrgb&w=2400",
                             **_fit(p["width"], p["height"], max_width=2400)}]}
            for p in data.get("photos", [])]


# ---------------------------------------------------------------- pixabay
def _pixabay(query, kind, orientation, per_page):
    params = {"key": os.environ[KEYS["pixabay"]], "q": query[:100], "per_page": max(3, per_page),
              "safesearch": "true"}
    if kind == "video":
        # No orientation filter for videos: the frame crop and the choice take care of it.
        # Only filmed footage: "animation" is mostly cheap motion graphics and green screens.
        data = _get("https://pixabay.com/api/videos/", params | {"video_type": "film"})
        return [{"provider": "pixabay", "id": h["id"], "kind": "video", "page_url": h["pageURL"],
                 "author": h.get("user", ""), "duration": h["duration"],
                 "width": h["videos"]["medium"]["width"] or h["videos"]["small"]["width"],
                 "height": h["videos"]["medium"]["height"] or h["videos"]["small"]["height"],
                 "ai_generated": bool(h.get("isAiGenerated")),
                 "thumb": h["videos"]["small"].get("thumbnail") or h["videos"]["tiny"].get("thumbnail", ""),
                 "renditions": [{"url": r["url"], "width": r["width"], "height": r["height"]}
                                for r in h["videos"].values() if r.get("url")]}
                for h in data.get("hits", [])]
    params |= {"image_type": "photo", "orientation": "vertical" if orientation == "portrait" else "horizontal"}
    data = _get("https://pixabay.com/api/", params)
    # largeImageURL is at most 1280 px on its long side (the full size needs extra API approval)
    return [{"provider": "pixabay", "id": h["id"], "kind": "photo", "page_url": h["pageURL"],
             "author": h.get("user", ""), "width": h["imageWidth"], "height": h["imageHeight"],
             "ai_generated": bool(h.get("isAiGenerated")), "thumb": h["webformatURL"],
             "renditions": [{"url": h["largeImageURL"], **_fit(h["imageWidth"], h["imageHeight"], long_side=1280)}]}
            for h in data.get("hits", [])]


SEARCHERS = {"pexels": _pexels, "pixabay": _pixabay}


def search(query, kind, orientation, providers, per_page=10):
    """kind: video | photo. orientation: landscape | portrait. Results of every provider with a key,
    in the given order. A provider that fails is skipped (with a message): the others still count."""
    results = []
    for provider in available(providers):
        try:
            results += SEARCHERS[provider](query, kind, orientation, per_page)
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"  · {provider}: search failed ({query!r}, {kind}): {e}")
    for r in results:            # sizes are needed to choose: a rendition without them is useless
        r["renditions"] = [x for x in r["renditions"] if x.get("width") and x.get("height")]
    return [r for r in results if r["renditions"] and r.get("thumb") and not r["ai_generated"]]


# --------------------------------------------------------------- download
def thumbnail(candidate, folder):
    """The candidate's preview image (a frame for videos) -> local file (kept)."""
    path = Path(folder) / f"{candidate['provider']}_{candidate['id']}_thumb.jpg"
    if not path.exists():
        r = requests.get(candidate["thumb"], headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def upscale_needed(rendition, width, height):
    """How much a rendition must be enlarged to fill (cover) a width x height frame."""
    return max(width / rendition["width"], height / rendition["height"])


def best_rendition(candidate, width, height):
    """The smallest rendition that fills the frame without enlarging more than MAX_UPSCALE;
    if none does, the largest."""
    renditions = sorted(candidate["renditions"], key=lambda r: r["width"] * r["height"])
    for r in renditions:
        if upscale_needed(r, width, height) <= MAX_UPSCALE:
            return r
    return renditions[-1]


def download(candidate, width, height, folder):
    """Downloads the candidate to folder/<provider>_<id>.<ext> (kept: a second run does not download again)."""
    rendition = best_rendition(candidate, width, height)
    ext = ".mp4" if candidate["kind"] == "video" else Path(rendition["url"].split("?")[0]).suffix.lower() or ".jpg"
    path = Path(folder) / f"{candidate['provider']}_{candidate['id']}_{rendition['width']}{ext}"
    if path.exists() and path.stat().st_size > 0:
        return path
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(rendition["url"], stream=True, timeout=TIMEOUT, headers=HEADERS) as r:
        r.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    partial.replace(path)
    return path
