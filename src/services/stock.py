"""Free footage and photos with a licence to use them. Sources of the visuals node (graph/nodes/visuals.py).
    pexels, pixabay   stock banks: videos and photos
    nasa              NASA Image and Video Library: photos of space, missions, the Earth from orbit
                      (public domain). Its videos are skipped: they are edited pieces (narration,
                      logos, titles), not footage to cut.
    wikimedia         Wikimedia Commons: photos of real, named things (companies, buildings, people,
                      places). Only free licences that allow commercial use (public domain, CC0,
                      CC BY, CC BY-SA): they ask for attribution, kept in the candidate's "license".

    search(query, kind, orientation, providers)   -> [candidate, ...] in the providers' order
    thumbnail(candidate, folder)                   -> local JPEG (to compare it with the text)
    download(candidate, width, height, folder)     -> local file, in the smallest size that fills the frame

A candidate is a plain dict, the same for every provider:
    {"provider", "id", "kind": "video" | "photo", "page_url", "caption", "author", "license", "width",
     "height", "thumb", "duration" (videos, s), "ai_generated", "renditions": [{"url", "width", "height"}, ...]}
caption: what its source says it is (NASA's title and description, the Commons file name, Pixabay's
tags, Pexels' alt text or page name), for the vision judge: a NASA "Dawn spacecraft in a thermal
vacuum chamber" is that, however much it looks like a physics lab.

Keys in the .env: PEXELS_API_KEY, PIXABAY_API_KEY (NASA and Wikimedia need none). A provider
without its key is skipped. Rate limits: Pixabay 100 requests/min, Pexels 200/hour.
"""
import html
import os
import re
import time
from pathlib import Path

import requests

from llmkit import config

TIMEOUT = 20
MAX_UPSCALE = 1.35        # a rendition may be enlarged this much to fill the frame (more looks blurry)
MAX_WAIT = 120            # seconds waited at most after a 429 before trying again
KEYS = {"pexels": "PEXELS_API_KEY", "pixabay": "PIXABAY_API_KEY", "nasa": None, "wikimedia": None}
HEADERS = {"User-Agent": "Mozilla/5.0"}
# Wikimedia asks for a User-Agent that says who is calling (a generic one may be blocked)
PROVIDER_HEADERS = {"wikimedia": {"User-Agent": config.USER_AGENT}}
# Seconds between file requests (thumbnails, downloads): Wikimedia answers 429 to quick bursts
PACE = {"wikimedia": 1.0}
_last_file_request = {}


def available(providers):
    """The providers (in order) that have a key in the .env (or need none)."""
    return [p for p in providers if KEYS[p] is None or os.getenv(KEYS[p], "").strip()]


def _request(url, params=None, headers=None, retries=2, stream=False):
    """GET with a wait-and-retry on 429 (rate limit) -> the response."""
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT, stream=stream)
        if r.status_code == 429 and attempt < retries:
            r.close()
            wait = int(r.headers.get("X-RateLimit-Reset") or r.headers.get("Retry-After") or 30)
            if wait > 10 ** 9:                  # Pexels sends a timestamp (the end of its period), not seconds
                wait -= time.time()
            # at least 10 s, 20 s...: Wikimedia's reset is often too short for a burst to clear
            time.sleep(min(max(wait, 10 * (attempt + 1)), MAX_WAIT) + 1)
            continue
        r.raise_for_status()
        return r


def _fetch_file(candidate, url, stream=False):
    """GET of one of the candidate's files, spaced as its provider asks (PACE)."""
    provider = candidate["provider"]
    pause = PACE.get(provider, 0) - (time.time() - _last_file_request.get(provider, 0))
    if pause > 0:
        time.sleep(pause)
    try:
        return _request(url, headers=PROVIDER_HEADERS.get(provider, HEADERS), retries=3, stream=stream)
    finally:
        _last_file_request[provider] = time.time()


def _get(url, params=None, headers=None, retries=2):
    """GET of a JSON API, with a wait-and-retry on 429."""
    return _request(url, params, headers, retries).json()


CAPTION_CHARS = 200       # of a NASA description (its first sentences say what the photo is)


def _slug(page_url):
    """The words of a page's name: '.../video/laser-beam-in-a-lab-6079428/' -> 'laser beam in a lab'."""
    name = page_url.rstrip("/").split("/")[-1]
    return re.sub(r"-?\d+$", "", name).replace("-", " ").strip()


def _fit(width, height, max_width=None, long_side=None):
    scale = min(1.0, max_width / width) if max_width else min(1.0, long_side / max(width, height))
    return {"width": round(width * scale), "height": round(height * scale)}


# ----------------------------------------------------------------- pexels
def _pexels(query, kind, orientation, per_page):
    headers = {"Authorization": os.environ[KEYS["pexels"]]}
    params = {"query": query, "orientation": orientation, "per_page": per_page}
    if kind == "video":
        data = _get("https://api.pexels.com/videos/search", params, headers)
        return [{"provider": "pexels", "id": v["id"], "kind": "video", "page_url": v["url"], "caption": _slug(v["url"]),
                 "author": (v.get("user") or {}).get("name", ""), "license": "Pexels License",
                 "width": v["width"], "height": v["height"],
                 "duration": v["duration"], "ai_generated": False, "thumb": v.get("image", ""),
                 "renditions": [{"url": f["link"], "width": f["width"], "height": f["height"]}
                                for f in v["video_files"] if f.get("file_type") == "video/mp4"]}
                for v in data.get("videos", [])]
    data = _get("https://api.pexels.com/v1/search", params, headers)
    return [{"provider": "pexels", "id": p["id"], "kind": "photo", "page_url": p["url"],
             "caption": p.get("alt") or _slug(p["url"]),
             "author": p.get("photographer", ""), "license": "Pexels License",
             "width": p["width"], "height": p["height"],
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
        return [{"provider": "pixabay", "id": h["id"], "kind": "video", "page_url": h["pageURL"], "caption": h.get("tags", ""),
                 "author": h.get("user", ""), "license": "Pixabay Content License", "duration": h["duration"],
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
    return [{"provider": "pixabay", "id": h["id"], "kind": "photo", "page_url": h["pageURL"], "caption": h.get("tags", ""),
             "author": h.get("user", ""), "license": "Pixabay Content License",
             "width": h["imageWidth"], "height": h["imageHeight"],
             "ai_generated": bool(h.get("isAiGenerated")), "thumb": h["webformatURL"],
             "renditions": [{"url": h["largeImageURL"], **_fit(h["imageWidth"], h["imageHeight"], long_side=1280)}]}
            for h in data.get("hits", [])]


# ------------------------------------------------------------------- nasa
def _nasa(query, kind, orientation, per_page):
    if kind == "video":
        return []
    data = _get("https://images-api.nasa.gov/search", {"q": query, "media_type": "image", "page_size": per_page})
    found = []
    for item in data.get("collection", {}).get("items", []):
        info, links = item["data"][0], [x for x in item.get("links", []) if x.get("render") == "image"]
        sized = [{"url": x["href"], "width": x["width"], "height": x["height"]}
                 for x in links if x.get("width") and x.get("height")]
        if not sized:
            continue
        preview = next((x["href"] for x in links if x.get("rel") == "preview"), min(sized, key=lambda r: r["width"])["url"])
        largest = max(sized, key=lambda r: r["width"])
        found.append({"provider": "nasa", "id": re.sub(r"[^\w.-]", "_", info["nasa_id"]), "kind": "photo",
                      "page_url": "https://images.nasa.gov/details/" + requests.utils.quote(info["nasa_id"]),
                      "caption": f"{info.get('title', '')}. {(info.get('description') or '')[:CAPTION_CHARS]}".strip(". "),
                      "author": info.get("photographer") or info.get("secondary_creator") or f"NASA {info.get('center', '')}".strip(),
                      "license": "NASA (dominio público)", "width": largest["width"], "height": largest["height"],
                      "ai_generated": False, "thumb": preview, "renditions": sized})
    return found


# -------------------------------------------------------------- wikimedia
FREE_LICENSE = re.compile(r"^(public domain|pd\b|cc0|cc by(-sa)? [\d.]+)", re.IGNORECASE)
# Thumbnail widths Commons serves (it refuses most others). Its thumbnails come from
# thumb.wikimedia.org; upload.wikimedia.org (the originals) answers 429 after one or two files.
COMMONS_THUMB = 500
COMMONS_SIZES = (960, 1280, 1920, 3840)


def _commons_renditions(info):
    """The standard-width thumbnails smaller than the original; the original only without any."""
    thumb, marker = info.get("thumburl", ""), f"/{COMMONS_THUMB}px-"
    renditions = [{"url": thumb.replace(marker, f"/{w}px-"), "width": w,
                   "height": round(info["height"] * w / info["width"])}
                  for w in COMMONS_SIZES if marker in thumb and info["width"] > w]
    return renditions or [{"url": info["url"], "width": info["width"], "height": info["height"]}]


def _commons_author(meta):
    """Artist comes as HTML (a link to the user's page): only its text."""
    text = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", ""))
    return html.unescape(text).strip()


def _wikimedia(query, kind, orientation, per_page):
    if kind == "video":
        return []
    params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
              "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": per_page,
              "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": COMMONS_THUMB,
              "iiextmetadatafilter": "LicenseShortName|Artist|NonFree"}
    data = _get("https://commons.wikimedia.org/w/api.php", params, PROVIDER_HEADERS["wikimedia"])
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    found = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        licence = (meta.get("LicenseShortName") or {}).get("value", "")
        if (info.get("mime") not in ("image/jpeg", "image/png", "image/webp") or (meta.get("NonFree") or {}).get("value")
                or max(info.get("width", 0), info.get("height", 0)) < 1280      # too small for a full frame
                or not FREE_LICENSE.match(licence) or re.search(r"\b(nc|nd)\b", licence, re.IGNORECASE)):
            continue
        found.append({"provider": "wikimedia", "id": page["pageid"], "kind": "photo",
                      "page_url": info.get("descriptionurl", ""), "author": _commons_author(meta),
                      "caption": re.sub(r"\.\w+$", "", page.get("title", "").removeprefix("File:")).replace("_", " "),
                      "license": licence, "width": info["width"], "height": info["height"], "ai_generated": False,
                      "thumb": info.get("thumburl") or info["url"], "renditions": _commons_renditions(info)})
    return found


SEARCHERS = {"pexels": _pexels, "pixabay": _pixabay, "nasa": _nasa, "wikimedia": _wikimedia}


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
        path.write_bytes(_fetch_file(candidate, candidate["thumb"]).content)
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
    with _fetch_file(candidate, rendition["url"], stream=True) as r:
        with open(partial, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    partial.replace(path)
    return path
