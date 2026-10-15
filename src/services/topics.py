"""Topic ideas for the channel from the RSS feeds of config.yaml (topics.feeds), and the record of
every topic (data/temas/registro.json) so that none is made twice. You choose; the model only proposes.

    1. read the feeds (no LLM): recent entries, without skip_domains, at most max_per_feed each,
       and without the entries already in the record (proposed, discarded, made or uploaded)
    2. one call to the model (llm.topics, prompts/topics.md): it groups the entries that tell the
       same story and scores them for the channel (a surprising number, can it be explained,
       does it last, is there something to show). They go to the record as "propuesto"
    3. the list: the proposed topics of the last reserve_days, best first (the new ones and the
       ones you did not choose yet). data/temas/<yyyymmdd-hhmm>.md has it with a command each

Used by the pipeline (graph/nodes/topics.py: run_pipeline.py without --topic -> choose_topic ⏸)
and by scripts/find_topics.py (only the list).

The record, one entry per topic:
    {"id", "topic", "status", "sources": [{"title", "url", "site"}], "datazo", "missing", "hook",
     "type", "format", "visuals", "score", "why", "found", "updated", "run_id", "youtube"}
    status: propuesto -> en_curso (a run started with it: run_id) -> subido (youtube: the link)
            descartado: you said no; it is never proposed again
A run started with --topic is recorded too (en_curso), so its articles are not proposed again.

A feed can be the feed itself or the site's home page: then the feed is looked for in its
<link rel="alternate" type="application/rss+xml">. A feed that cannot be read is reported and skipped.
"""
import hashlib
import html
import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from urllib.parse import urljoin

import feedparser
import requests
from pydantic import BaseModel, Field

from llmkit import config
from prompts import load_prompt
from services.articles import HEADERS, blocked
from services.settings import load_settings

NODE = "topics"
SUMMARY_CHARS = 280             # of each entry, for the model: the title is what matters
FEED_LINK = re.compile(r"<link[^>]+type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*>", re.I)
HREF = re.compile(r"href=[\"']([^\"']+)[\"']", re.I)
TAG = re.compile(r"<[^>]+>")
STATUS_LABELS = {"propuesto": "propuesto", "en_curso": "en curso", "subido": "subido", "descartado": "descartado"}


class TopicCandidate(BaseModel):
    topic: str = Field(description="El tema del vídeo en una frase clara, en español (lo que va en --topic)")
    entries: list[int] = Field(min_length=1, max_length=6,
                               description="Números de las entradas que cuentan esta historia (la mejor fuente primero)")
    datazo: str = Field(description="El dato que sorprende, tal como lo dan las entradas (con su cifra si la dan); "
                                    "vacío si las entradas no traen ninguno. Nada que no esté en ellas")
    missing: str = Field(description="Lo que habría que buscar porque las entradas no lo dan (la cifra exacta, "
                                     "el mecanismo...); vacío si no falta nada")
    hook: str = Field(description="La frase con que empezaría el vídeo")
    type: Literal["news", "explainer"] = Field(description="news: solo vale ahora; explainer: la noticia es el "
                                                           "gancho y el vídeo explica algo que no caduca")
    format: Literal["short", "long"] = Field(description="short: se explica en algo más de un minuto; long: pide ~5 min de contexto")
    visuals: str = Field(description="Qué se podría enseñar (fotos reales, NASA, figuras de un paper, Wikimedia...)")
    score: int = Field(ge=1, le=10, description="1-10 para el canal")
    why: str = Field(description="Por qué esa nota, en una frase: lo bueno y lo que falla")


class TopicsDraft(BaseModel):
    candidates: list[TopicCandidate] = Field(max_length=20, description="Los mejores temas, de mayor a menor nota")


# ------------------------------------------------------------------ feeds
def _clean(text):
    return " ".join(html.unescape(TAG.sub(" ", text or "")).split())


def _when(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime(*parsed[:6], tzinfo=timezone.utc) if parsed else None


def _parse(url, timeout):
    """-> (feedparser result, the feed's url). A home page -> the feed it links to."""
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    if feed.entries:
        return feed, url
    for link in FEED_LINK.findall(r.text):            # a web page: its own feed
        href = HREF.search(link)
        if href:
            feed_url = urljoin(url, html.unescape(href.group(1)))
            r = requests.get(feed_url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            feed = feedparser.parse(r.content)
            if feed.entries:
                return feed, feed_url
    return feed, url


def read_feeds(feeds, days=7, max_per_feed=15, skip_domains=(), timeout=15):
    """-> (entries, problems). entries: {"title", "summary", "url", "date", "site", "kind"}, newest first.
    A feed: its url, or {"url", "name", "kind", "skip": [pieces of url to leave out]}."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    entries, problems, seen = [], [], set()
    for feed_conf in feeds:
        feed_conf = {"url": feed_conf} if isinstance(feed_conf, str) else feed_conf
        try:
            feed, feed_url = _parse(feed_conf["url"], timeout)
        except requests.RequestException as e:
            problems.append(f"{feed_conf['url']}: {e.__class__.__name__}")
            continue
        if not feed.entries:
            problems.append(f"{feed_conf['url']}: sin entradas (¿no es un feed?)")
            continue
        site = feed_conf.get("name") or _clean(feed.feed.get("title")) or feed_url
        kept = 0
        for e in feed.entries:
            url, when = e.get("link"), _when(e)
            if not url or url in seen or blocked(url, skip_domains):
                continue
            if any(part in url for part in feed_conf.get("skip", [])):     # e.g. "/seleccion/": shop deals
                continue
            if when and when < since:                  # an entry without a date is kept
                continue
            seen.add(url)
            summary = _clean(e.get("summary"))
            entries.append({"title": _clean(e.get("title")), "url": url, "site": site,
                            "kind": feed_conf.get("kind", ""),
                            "date": when.date().isoformat() if when else "",
                            "summary": summary[:SUMMARY_CHARS] + ("…" if len(summary) > SUMMARY_CHARS else "")})
            kept += 1
            if kept >= max_per_feed:
                break
    entries.sort(key=lambda x: x["date"], reverse=True)
    return entries, problems


# ------------------------------------------------------------------ the record
def _options(settings=None):
    return (settings or load_settings()).get("topics", {})


def folder(settings=None):
    path = config.BASE / _options(settings).get("out_dir", "data/temas")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(settings=None):
    path = folder(settings) / "registro.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_registry(settings=None):
    return _read(settings).get("topics", [])


def save_registry(topics, settings=None, seen=None):
    """seen: {url: date} of the entries the model already read (proposed or not), so they are not
    sent again; None keeps the saved ones."""
    data = _read(settings)
    data["topics"] = topics
    if seen is not None:
        data["seen"] = seen
    (folder(settings) / "registro.json").write_text(
        json.dumps({"topics": data["topics"], "seen": data.get("seen", {})}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def topic_id(urls, topic=""):
    """The same first article -> the same id (a run with --topic and no --url: by the topic)."""
    return hashlib.sha1((urls[0] if urls else topic.lower().strip()).encode("utf-8")).hexdigest()[:8]


def known_urls(topics):
    return {s["url"] for t in topics for s in t.get("sources", [])}


def _set(topics, item_id, **fields):
    for t in topics:
        if t["id"] == item_id:
            t.update(fields, updated=date.today().isoformat())
            return t
    return None


def discard(item_ids, settings=None):
    topics = load_registry(settings)
    for item_id in item_ids:
        _set(topics, item_id, status="descartado")
    save_registry(topics, settings)


def start_run(run_id, item=None, topic="", urls=(), settings=None):
    """A run starts: the chosen topic (item) or yours (--topic, --url) -> en_curso."""
    topics = load_registry(settings)
    if item is None:
        item = {"id": topic_id(list(urls), topic), "topic": topic, "sources": [{"url": u} for u in urls],
                "found": date.today().isoformat()}
        if not any(t["id"] == item["id"] for t in topics):
            topics.append(item)
    _set(topics, item["id"], status="en_curso", run_id=run_id)
    save_registry(topics, settings)


def mark_uploaded(run_id, link, settings=None):
    """upload_youtube.py: the run's topic -> subido. False if the run is not in the record."""
    topics = load_registry(settings)
    found = next((t for t in topics if t.get("run_id") == run_id), None)
    if found:
        _set(topics, found["id"], status="subido", youtube=link)
        save_registry(topics, settings)
    return bool(found)


def done_titles(topics, settings=None):
    """What the model must not propose again: the topics of the record that are not just proposed,
    the first line of every run's script.md and the folders prepared or uploaded to YouTube (the
    videos made before the record existed)."""
    settings = settings or load_settings()
    titles = [t["topic"] for t in topics if t.get("status") != "propuesto"]
    for md in (config.STATE_DIR / "runs").glob("*/script.md"):
        first = md.read_text(encoding="utf-8").split("\n", 1)[0]
        if first.startswith("# "):
            titles.append(first[2:].strip())
    youtube = settings.get("youtube", {})
    for key, default in (("ready_dir", "data/listo_para_subir"), ("uploaded_dir", "data/subidos")):
        for path in (config.BASE / youtube.get(key, default)).glob("*"):
            if path.is_dir():
                titles.append(path.name.split("_", 1)[-1].replace("_", " "))
    return sorted(set(titles))


def reserve(topics, days, skip=()):
    """The proposed topics found in the last `days` days, best first."""
    since = (date.today() - timedelta(days=days)).isoformat()
    ready = [t for t in topics if t.get("status") == "propuesto" and t.get("found", "") >= since
             and t["id"] not in skip]
    return sorted(ready, key=lambda t: (t.get("score", 0), t.get("found", "")), reverse=True)


# ------------------------------------------------------------------ the model
def topics_request(entries, done, how_many):
    parts = [f"Propón como mucho {how_many} temas.", "", "# Entradas recientes"]
    for i, e in enumerate(entries, 1):
        parts.append(f"[{i}] {e['title']} ({e['site']}, {e['date'] or 'sin fecha'})")
        if e["summary"]:
            parts.append(f"    {e['summary']}")
    if done:
        parts += ["", "# Vídeos ya hechos o descartados (no los propongas otra vez, ni el mismo tema contado "
                      "de otra forma)"]
        parts += [f"- {t}" for t in done]
    return "\n".join(parts)


def score(entries, hub, done=(), how_many=10):
    """-> new record items (status propuesto), best first."""
    draft = hub.chain(load_prompt(NODE), schema=TopicsDraft, node=NODE).invoke(
        {"text": topics_request(entries, done, how_many)})
    items = []
    for c in sorted(draft.candidates, key=lambda c: c.score, reverse=True)[:how_many]:
        numbers = [n for n in dict.fromkeys(c.entries) if 1 <= n <= len(entries)]
        if not numbers:
            continue
        sources = [{k: entries[n - 1][k] for k in ("title", "url", "site")} for n in numbers]
        items.append({"id": topic_id([s["url"] for s in sources]), "topic": c.topic, "status": "propuesto",
                      "sources": sources, **c.model_dump(exclude={"topic", "entries"}),
                      "found": date.today().isoformat()})
    return items


# ------------------------------------------------------------------ the file
def _quote(text):
    return '"' + text.replace('"', "'") + '"'


def notes_of(item):
    return f"El datazo: {item['datazo']}" if item.get("datazo") else ""


def command(item):
    parts = ["python scripts/run_pipeline.py", "--topic", _quote(item["topic"]),
             "--type", item["type"], "--format", item["format"]]
    if notes_of(item):
        parts += ["--notes", _quote(notes_of(item))]
    for s in item["sources"]:
        parts += ["--url", s["url"]]
    return " ".join(parts)


def short_line(n, item):
    kind = "explicativo" if item["type"] == "explainer" else "noticia"
    sites = ", ".join(dict.fromkeys(s.get("site", "") for s in item["sources"]))
    return f"[{n}] {item['score']}/10 {item['topic']} · {kind} · {item['format']} · {sites}"


def markdown(items, n_entries, problems, days):
    lines = [f"# Temas para el canal · {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{n_entries} entradas nuevas de los últimos {days} días (las que ya están en registro.json no "
             f"se puntúan otra vez); abajo, los temas propuestos aún sin hacer.", ""]
    for i, t in enumerate(items, 1):
        kind = "explicativo" if t["type"] == "explainer" else "noticia"
        lines += [f"## {i}. [{t['score']}/10] {t['topic']}", "",
                  f"- **Datazo:** {t.get('datazo') or '—'}",
                  f"- **Falta:** {t.get('missing') or '—'}",
                  f"- **Gancho:** {t['hook']}",
                  f"- **Tipo:** {kind} · {t['format']} · **Imágenes:** {t['visuals']}",
                  f"- **Por qué:** {t['why']}",
                  f"- **Encontrado:** {t['found']} · id {t['id']}",
                  "- **Fuentes:**"]
        lines += [f"  - [{s['title']}]({s['url']}) ({s['site']})" for s in t["sources"]]
        lines += ["", "```", command(t), "```", ""]
    if problems:
        lines += ["## Feeds que no se pudieron leer", ""] + [f"- {p}" for p in problems] + [""]
    return "\n".join(lines)


def entries_markdown(entries, problems, days):
    """Without the model: just the entries, to read them yourself."""
    lines = [f"# Entradas · {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{len(entries)} entradas de los últimos {days} días.", ""]
    lines += [f"- {e['date'] or '?'} · [{e['title']}]({e['url']}) ({e['site']})" for e in entries]
    if problems:
        lines += ["", "## Feeds que no se pudieron leer", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def find(hub=None, days=None, how_many=None, extra_feeds=(), use_model=True, skip=()):
    """-> (path of the .md, the topics to choose from). skip: ids not to show (already seen).
    hub=None with use_model=False: only the entries, no LLM and no record."""
    settings = load_settings()
    options = _options(settings)
    days = days or options.get("days", 7)
    how_many = how_many or options.get("max_candidates", 10)
    entries, problems = read_feeds(list(options.get("feeds", [])) + list(extra_feeds), days=days,
                                   max_per_feed=options.get("max_per_feed", 15),
                                   skip_domains=settings.get("research", {}).get("skip_domains", []))
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    if not use_model:
        path = folder(settings) / f"{stamp}_entradas.md"
        path.write_text(entries_markdown(entries, problems, days), encoding="utf-8")
        return path, []
    topics = load_registry(settings)
    # Entries read before (proposed or not) are not scored again; forgotten after twice the window
    since = (date.today() - timedelta(days=2 * days)).isoformat()
    seen = {url: when for url, when in _read(settings).get("seen", {}).items() if when >= since}
    known = known_urls(topics) | set(seen)
    new = [e for e in entries if e["url"] not in known][:options.get("max_entries", 250)]
    if new:
        print(f"  · topics: {len(new)} entradas nuevas de {len(entries)} -> el modelo las puntúa")
        ids = {t["id"] for t in topics}
        topics += [t for t in score(new, hub, done_titles(topics, settings), how_many) if t["id"] not in ids]
        seen.update({e["url"]: date.today().isoformat() for e in new})
        save_registry(topics, settings, seen)
    else:
        print(f"  · topics: ninguna entrada nueva ({len(entries)} ya están en el registro): sin llamar al modelo")
    items = reserve(topics, options.get("reserve_days", 30), skip)[:how_many]
    path = folder(settings) / f"{stamp}.md"
    path.write_text(markdown(items, len(new), problems, days), encoding="utf-8")
    return path, items
