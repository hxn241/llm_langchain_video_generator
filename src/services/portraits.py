"""Portraits for the persona labels (rótulos, graph/nodes/labels.py): a photo of the person named, from
Wikidata + Wikimedia Commons, only when it is surely them.

    candidates(name)                       -> people on Wikidata with EXACTLY that name and a photo
    download(candidate, folder)            -> the photo (free licence only) and its credit, or None

The same name is not enough (namesakes: a footballer with a researcher's name): labels asks an LLM
whether one of the candidates is the person the video talks about (their occupations, employers and
years against the label's role and the narration), and when in doubt there is no photo.
Wikidata's "image" (P18) is always a Commons file; Commons only holds free files, but the licence is
checked again (no NC/ND), as the stock search does.
"""
import re
import unicodedata

from PIL import Image

from llmkit import config
from services import stock

WIKIDATA = "https://www.wikidata.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": config.USER_AGENT}      # Wikimedia blocks generic ones
HUMAN = "Q5"
LANGUAGES = "es|en|mul"
SEARCH_LIMIT = 7                                 # per language
# A thumbnail width Commons serves (it refuses most others), bigger than the frame's photo (~170-220 px)
# and smaller than most originals: a file smaller than it comes from upload.wikimedia.org, which answers 429
PHOTO_WIDTH = 330


def plain(text):
    """'José  Martínez-Ruiz' -> 'jose martinez ruiz': the name without accents, case or punctuation."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _get(url, params):
    return stock._request(url, {**params, "format": "json"}, HEADERS, retries=3).json()


def _texts(entity, field):
    """Every label / alias / description of an entity, in LANGUAGES."""
    values = (entity.get(field) or {}).values()
    return [v["value"] for item in values for v in (item if isinstance(item, list) else [item])]


def _first_text(entity, field):
    for language in LANGUAGES.split("|"):
        value = (entity.get(field) or {}).get(language)
        if value:
            return value["value"]
    return ""


def _claims(entity, prop):
    """The values of a property, best rank first, without deprecated ones."""
    claims = [c for c in (entity.get("claims") or {}).get(prop, []) if c.get("rank") != "deprecated"]
    claims.sort(key=lambda c: c.get("rank") != "preferred")
    return [c["mainsnak"]["datavalue"]["value"] for c in claims if c["mainsnak"].get("datavalue")]


def _year(entity, prop):
    for value in _claims(entity, prop):
        found = re.match(r"[+-]?(\d{1,4})-", value.get("time", ""))
        if found:
            return int(found.group(1))
    return None


def _entities(ids, props):
    out = {}
    for n in range(0, len(ids), 50):
        data = _get(WIKIDATA, {"action": "wbgetentities", "ids": "|".join(ids[n:n + 50]), "props": props,
                               "languages": LANGUAGES})
        out.update(data.get("entities") or {})
    return out


def candidates(name):
    """People on Wikidata whose label or alias is exactly `name` (accents and case aside) and who have a
    photo: [{"id", "name", "description", "occupations", "employers", "born", "died", "image"}]."""
    wanted, ids = plain(name), []
    if not wanted:
        return []
    for language in ("es", "en"):
        data = _get(WIKIDATA, {"action": "wbsearchentities", "search": name, "language": language,
                               "uselang": language, "type": "item", "limit": SEARCH_LIMIT})
        ids += [r["id"] for r in data.get("search", []) if r["id"] not in ids]
    if not ids:
        return []
    people = []
    for qid, entity in _entities(ids, "labels|aliases|descriptions|claims").items():
        is_human = HUMAN in [v.get("id") for v in _claims(entity, "P31")]
        same_name = wanted in {plain(t) for t in _texts(entity, "labels") + _texts(entity, "aliases")}
        images = _claims(entity, "P18")
        if is_human and same_name and images:
            people.append({"id": qid, "entity": entity, "image": images[0]})
    if not people:
        return []
    linked = {v["id"] for p in people for prop in ("P106", "P108") for v in _claims(p["entity"], prop)}
    names = {qid: _first_text(e, "labels") for qid, e in _entities(sorted(linked), "labels").items()} if linked else {}
    return [{"id": p["id"], "name": _first_text(p["entity"], "labels"),
             "description": _first_text(p["entity"], "descriptions"),
             "occupations": [names.get(v["id"], "") for v in _claims(p["entity"], "P106") if names.get(v["id"])],
             "employers": [names.get(v["id"], "") for v in _claims(p["entity"], "P108") if names.get(v["id"])],
             "born": _year(p["entity"], "P569"), "died": _year(p["entity"], "P570"), "image": p["image"]}
            for p in people]


def download(candidate, folder):
    """The candidate's photo -> <folder>/<name>.jpg and its credit, or None (not a free licence, not a
    photo format, no thumbnail): {"path", "author", "license", "page_url"}."""
    data = _get(COMMONS, {"action": "query", "titles": f"File:{candidate['image']}", "prop": "imageinfo",
                          "iiprop": "url|size|mime|extmetadata", "iiurlwidth": PHOTO_WIDTH,
                          "iiextmetadatafilter": "LicenseShortName|Artist|NonFree"})
    page = next(iter((data.get("query") or {}).get("pages", {}).values()), {})
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})
    licence = (meta.get("LicenseShortName") or {}).get("value", "")
    if (info.get("mime") not in ("image/jpeg", "image/png", "image/webp") or (meta.get("NonFree") or {}).get("value")
            or not stock.FREE_LICENSE.match(licence) or re.search(r"\b(nc|nd)\b", licence, re.IGNORECASE)):
        return None
    url = info.get("thumburl") or info.get("url")
    if not url:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"{plain(candidate['name']).replace(' ', '_') or candidate['id']}_{candidate['id']}.download"
    raw.write_bytes(stock._request(url, headers=HEADERS, retries=3).content)
    path = raw.with_suffix(".jpg")
    with Image.open(raw) as image:
        image.convert("RGB").save(path, quality=92)
    raw.unlink()
    author = stock._commons_author(meta)
    if author.lower().startswith("unknown author"):         # Commons' "author unknown" template, often twice
        author = "Autor desconocido"
    return {"path": path, "author": author, "license": licence,
            "page_url": info.get("descriptionurl", "")}
