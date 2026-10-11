"""Wikipedia articles as plain text, through its free API (no key), for deep_research: the
background of a long video (history, how something works, who is who, known cases).

    article("es:Energía geotérmica")   -> {"url", "title", "text", "date": ""} or None
    article("en:Enhanced geothermal system")
A name that is not an exact title is searched and the first result is taken.
relevant(text, question, max_chars) keeps the article's opening and the paragraphs that share
the most words with what is being looked for (a whole article is too long for one call).
"""
import re
import unicodedata

import requests

HEADERS = {"User-Agent": "llm-video-generator/0.1 (personal research tool; python-requests)"}
TIMEOUT = 20
STOPWORDS = {"de", "la", "el", "los", "las", "un", "una", "y", "o", "en", "a", "por", "para", "con", "que",
             "del", "al", "se", "su", "sus", "es", "como", "qué", "cómo", "cuál", "cuándo", "quién", "lo",
             "the", "of", "and", "in", "to", "is", "for", "on", "with", "how", "what", "who", "when", "by"}


def _api(lang, **params):
    r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=HEADERS, timeout=TIMEOUT,
                     params={"format": "json", "formatversion": 2, **params})
    r.raise_for_status()
    return r.json()


def article(name):
    """'es:Title' / 'en:Title' / 'Title' (Spanish) -> the article as plain text, or None."""
    lang, _, title = name.partition(":") if re.match(r"^[a-z]{2,3}:", name) else ("es", "", name)
    title = title.strip()
    if not title:
        return None
    pages = _api(lang, action="query", prop="extracts", explaintext=1, redirects=1, titles=title)["query"]["pages"]
    page = pages[0] if pages else {}
    if page.get("missing") or not page.get("extract"):
        found = _api(lang, action="query", list="search", srsearch=title, srlimit=1)["query"]["search"]
        if not found:
            return None
        pages = _api(lang, action="query", prop="extracts", explaintext=1, titles=found[0]["title"])["query"]["pages"]
        page = pages[0] if pages else {}
    if not page.get("extract"):
        return None
    url_title = page["title"].replace(" ", "_")
    return {"url": f"https://{lang}.wikipedia.org/wiki/{url_title}", "title": page["title"],
            "text": page["extract"], "date": "", "outlet": f"{lang}.wikipedia.org"}


def _words(text):
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return {w for w in re.findall(r"[a-z0-9]{3,}", plain) if w not in STOPWORDS}


def relevant(text, wanted, max_chars):
    """The opening paragraph plus the paragraphs sharing most words with `wanted`, in their
    original order, up to max_chars."""
    if len(text) <= max_chars:
        return text
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 80]
    if not paragraphs:
        return text[:max_chars]
    keys = _words(wanted)
    ranked = sorted(range(1, len(paragraphs)), key=lambda i: -len(keys & _words(paragraphs[i])))
    keep, size = {0}, len(paragraphs[0])
    for i in ranked:
        if size + len(paragraphs[i]) > max_chars:
            continue
        keep.add(i)
        size += len(paragraphs[i])
    return "\n\n".join(paragraphs[i] for i in sorted(keep))
