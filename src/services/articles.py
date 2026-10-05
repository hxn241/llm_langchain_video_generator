"""Articles from the web: download a page and keep only its text (trafilatura),
without menus, ads or comments. Used by the research node.
"""
import requests
import trafilatura

# A browser's User-Agent: many news sites reject requests without one
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
           "Accept-Language": "es-ES,es;q=0.9,en;q=0.6"}
MIN_CHARS = 600         # less than this is usually a paywall, a cookie wall or a video page


def fetch_article(url, timeout=15):
    """-> {"url", "title", "text", "date"} or None if it cannot be read (blocked, paywall, too short)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return None
    text = trafilatura.extract(r.text, url=url, include_comments=False, include_tables=False,
                               favor_precision=True)
    if not text or len(text) < MIN_CHARS:
        return None
    metadata = trafilatura.extract_metadata(r.text, default_url=url)
    return {"url": url, "text": text,
            "title": (metadata.title if metadata and metadata.title else ""),
            "date": (metadata.date if metadata and metadata.date else "")}
