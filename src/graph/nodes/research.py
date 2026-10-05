"""research 🤖: topic (+ your notes and URLs) -> sources + facts, each with its source.

1. The LLM writes a few news searches for the topic (different aspects).
2. Your URLs first, then the search results: each article is downloaded and reduced
   to its text (trafilatura). Paywalls, blocked pages and off-topic articles are skipped.
3. The LLM extracts facts from EACH article separately (small context, and every fact
   keeps its exact source). Opinions are kept only attributed ("Según X, ...").

The facts are the only data the outline and the script may use. Every article read
is saved in <run>/research/ to check where a fact came from.
"""
from graph.state import FactsDraft, SearchQueries
from llmkit.web_search import search
from prompts import load_prompt
from services.articles import fetch_article
from services.settings import load_settings, runs_dir

NODE = "research"


def queries_request(topic, notes, n):
    parts = [f"Tema del vídeo: {topic}"]
    if notes:
        parts.append(f"Notas del autor: {notes}")
    parts.append(f"Escribe {n} búsquedas.")
    return "\n".join(parts)


def facts_request(topic, notes, article, max_chars):
    parts = [f"Tema del vídeo: {topic}"]
    if notes:
        parts.append(f"Notas del autor: {notes}")
    parts += ["", f"# Artículo: {article.get('title') or '(sin título)'}",
              f"Fuente: {article['url']}" + (f" · {article['date']}" if article.get("date") else ""),
              "", article["text"][:max_chars]]
    return "\n".join(parts)


def make_research(hub, settings=None):
    settings = settings or load_settings()
    options = settings["research"]
    write_queries = hub.chain(load_prompt("research_queries"), schema=SearchQueries, node=NODE)
    extract_facts = hub.chain(load_prompt("research_facts"), schema=FactsDraft, node=NODE)

    def research(state, config):
        topic, notes = state["topic"], state.get("notes", "")
        video_format = state.get("format") or settings.get("format", "long")
        user_urls = state.get("urls", [])
        limit = max(options["max_articles"][video_format], len(user_urls))
        folder = runs_dir(state.get("run_id") or config["configurable"]["thread_id"]) / "research"
        folder.mkdir(exist_ok=True)

        queries = write_queries.invoke({"text": queries_request(topic, notes, options["queries"])}).queries
        candidates = [{"url": url} for url in user_urls]               # yours first, always read
        for query in queries[:options["queries"]]:
            try:
                candidates += search(query, options["results_per_query"], region=options["region"],
                                     news=True)["results"]
            except Exception as e:                                     # one failed search does not stop it
                print(f"  · search failed ({query!r}): {e}")

        sources, facts, seen = [], [], set()
        for candidate in candidates:
            if len(sources) >= limit:
                break
            url = candidate["url"]
            if url in seen:
                continue
            seen.add(url)
            article = fetch_article(url)
            if article is None:
                continue
            found = extract_facts.invoke({"text": facts_request(topic, notes, article,
                                                                options["max_article_chars"])}).facts
            if not found:                                              # off-topic
                continue
            n = len(sources) + 1
            (folder / f"{n:02d}.txt").write_text(f"{url}\n\n{article['text']}", encoding="utf-8")
            sources.append({"url": url, "title": article["title"] or candidate.get("title", ""),
                            "outlet": candidate.get("outlet", ""), "date": article["date"] or candidate.get("date", "")})
            facts += [{"fact": f.fact.strip(), "source_url": url} for f in found if f.fact.strip()]

        if not facts:
            raise RuntimeError(f"No information found about {topic!r}: try another wording, or give "
                               "articles with --url")
        return {"sources": sources, "facts": facts}

    return research
