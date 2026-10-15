"""research 🤖: topic (+ your notes and URLs) -> sources + facts, each with its source.

Two kinds of video (state["video_type"]: --type, or the LLM decides it from the topic):
- news:      something that has just happened. News searches, news articles.
- explainer: something that does not expire (how a thing works, why it is so, the pros and cons
             of a technique). News searches would bring a quote in a newspaper or a sales listing,
             not the explanation: the searches go to the general web, and the Wikipedia articles
             that explain the topic are read first.

1. The LLM decides the kind (unless you gave it), writes the searches and, for an explainer,
   names the Wikipedia articles.
2. Your URLs first, then Wikipedia, then the search results: each page is downloaded and reduced
   to its text (trafilatura). Paywalls, blocked pages and sites that are never a source (shops,
   listings, social networks, forums: research.skip_domains) are skipped, and so is a text the
   LLM finds nothing in (off-topic, an ad).
3. The LLM extracts facts from EACH text separately (small context, and every fact keeps its
   exact source). Opinions are kept only attributed ("Según X, ...").

The facts are the only data the outline and the script may use. Every text read is saved in
<run>/research/ to check where a fact came from.
"""
from urllib.parse import urlparse

from graph.script_files import TYPE_LABELS
from graph.state import FactsDraft, SearchPlan
from llmkit.web_search import search
from prompts import load_prompt
from services import wikipedia
from services.articles import blocked, fetch_article
from services.settings import load_settings, runs_dir

NODE = "research"


def queries_request(topic, notes, n, video_type=None):
    parts = [f"Tema del vídeo: {topic}"]
    if notes:
        parts.append(f"Notas del autor: {notes}")
    if video_type:
        parts.append(f"Tipo de vídeo: {video_type} (lo ha decidido el autor)")
    parts.append(f"Escribe {n} búsquedas.")
    return "\n".join(parts)


def facts_request(topic, notes, article, max_chars):
    parts = [f"Tema del vídeo: {topic}"]
    if notes:
        parts.append(f"Notas del autor: {notes}")
    parts += ["", f"# Texto: {article.get('title') or '(sin título)'}",
              f"Fuente: {article['url']}" + (f" · {article['date']}" if article.get("date") else ""),
              "", article["text"][:max_chars]]
    return "\n".join(parts)


def read(candidate, wanted, max_chars):
    """A candidate ({"url"} or {"wiki": "es:Title"}) -> its text, or None. Of a Wikipedia article,
    only its opening and the paragraphs about the topic (a whole one is too long for one call)."""
    try:
        if "wiki" in candidate:
            doc = wikipedia.article(candidate["wiki"])
            return doc and {**doc, "text": wikipedia.relevant(doc["text"], wanted, max_chars)}
        return fetch_article(candidate["url"])
    except Exception:                                                  # one bad page does not stop it
        return None


def make_research(hub, settings=None):
    settings = settings or load_settings()
    options = settings["research"]
    plan_search = hub.chain(load_prompt("research_queries"), schema=SearchPlan, node=NODE)
    extract_facts = hub.chain(load_prompt("research_facts"), schema=FactsDraft, node="facts")   # a cheaper model

    def research(state, config):
        topic, notes = state["topic"], state.get("notes", "")
        video_format = state.get("format") or settings.get("format", "long")
        user_urls = state.get("urls", [])
        limit = max(options["max_articles"][video_format], len(user_urls))
        folder = runs_dir(state.get("run_id") or config["configurable"]["thread_id"]) / "research"
        folder.mkdir(exist_ok=True)

        plan = plan_search.invoke({"text": queries_request(topic, notes, options["queries"], state.get("video_type"))})
        video_type = state.get("video_type") or plan.video_type
        news = video_type == "news"
        print(f"  · research: {TYPE_LABELS[video_type]}" + ("" if state.get("video_type") else
              "; lo decidió el modelo (para forzarlo: --type news o --type explainer)"))
        candidates = [{"url": url} for url in user_urls]               # yours first, always read
        if not news:
            candidates += [{"wiki": title} for title in plan.wikipedia]
        for query in plan.queries[:options["queries"]]:
            try:
                candidates += search(query, options["results_per_query"], region=options["region"],
                                     news=news)["results"]
            except Exception as e:                                     # one failed search does not stop it
                print(f"  · search failed ({query!r}): {e}")

        sources, facts, tried, done = [], [], set(), set()
        skip = options.get("skip_domains", [])
        for candidate in candidates:
            if len(sources) >= limit:
                break
            url = candidate.get("url")
            key = url or f"wiki:{candidate['wiki']}"
            if key in tried or (url and url not in user_urls and blocked(url, skip)):
                continue
            tried.add(key)
            article = read(candidate, f"{topic} {notes}", options["max_article_chars"])
            if article is None or article["url"] in done:              # e.g. a Wikipedia page found twice
                continue
            done.add(article["url"])
            found = extract_facts.invoke({"text": facts_request(topic, notes, article,
                                                                options["max_article_chars"])}).facts
            if not found:                                              # off-topic, or not a source
                continue
            n = len(sources) + 1
            (folder / f"{n:02d}.txt").write_text(f"{article['url']}\n\n{article['text']}", encoding="utf-8")
            sources.append({"url": article["url"], "title": article["title"] or candidate.get("title", ""),
                            "outlet": (candidate.get("outlet") or article.get("outlet")
                                       or urlparse(article["url"]).netloc.removeprefix("www.")),
                            "date": article["date"] or candidate.get("date", "")})
            facts += [{"fact": f.fact.strip(), "source_url": article["url"]} for f in found if f.fact.strip()]

        if not facts:
            raise RuntimeError(f"No information found about {topic!r}: try another wording, or give "
                               "articles with --url")
        return {"sources": sources, "facts": facts, "video_type": video_type}

    return research
