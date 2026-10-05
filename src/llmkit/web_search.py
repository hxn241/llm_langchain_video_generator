"""Web search. Chosen with SEARCH_PROVIDER in the .env.

- duckduckgo: a regular search engine, free, no key. 'content' is the snippet.
- tavily:     a search engine for agents (needs TAVILY_API_KEY). 'content' is the relevant passage.

Both return the SAME format, so the rest of the code does not change.

    search_web                       a ready-made tool with a general description
    create_search_tool(name, description, max_results, region)
                                     the same with your own description (tell the
                                     model WHEN to search the web in your product)
"""
import json
import os
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from llmkit import config

TAVILY_URL = "https://api.tavily.com/search"


def _iso_date(text):
    """Tavily's 'Wed, 30 Sep 2026 20:21:49 GMT' -> '2026-09-30' (as DuckDuckGo's); unknown formats as they come."""
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        return text


def search(query, max_results=3, provider=None, region=None, news=False):
    """-> {"answer": str | None, "results": [{"url", "title", "content"}, ...]}
    region: DuckDuckGo region (default SEARCH_REGION in the .env).
    news=True: recent news articles; each result also has "outlet" and "date"."""
    provider = provider or config.SEARCH_PROVIDER

    if provider == "duckduckgo":
        from ddgs import DDGS

        region = region or config.SEARCH_REGION
        if news:
            hits = DDGS().news(query, max_results=max_results, region=region)
            return {"answer": None, "results": [
                {"url": h["url"], "title": h["title"], "content": h["body"],
                 "outlet": h.get("source", ""), "date": h.get("date", "")} for h in hits]}
        hits = DDGS().text(query, max_results=max_results, region=region)
        return {
            "answer": None,
            "results": [{"url": h["href"], "title": h["title"], "content": h["body"]} for h in hits],
        }

    if provider == "tavily":
        # REST API (no SDK needed). Basic depth = 1 credit per search. region is not used:
        # the language of the query already picks the country's news.
        key = os.getenv("TAVILY_API_KEY", "").strip()
        if not key:
            raise RuntimeError("TAVILY_API_KEY is missing in the .env file")
        r = requests.post(TAVILY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30,
                          json={"query": query, "max_results": max_results, "include_answer": True,
                                "topic": "news" if news else "general", "search_depth": "basic"})
        if not r.ok:
            raise RuntimeError(f"Tavily search failed ({r.status_code}): {r.text[:300]}")
        response = r.json()
        results = [{"url": h.get("url"), "title": h.get("title"), "content": h.get("content")}
                   for h in response.get("results", [])]
        if news:
            for h, result in zip(response.get("results", []), results):
                result["outlet"] = urlparse(h.get("url") or "").netloc.removeprefix("www.")
                result["date"] = _iso_date(h.get("published_date", ""))
        return {"answer": response.get("answer"), "results": results}

    raise ValueError(f"Unknown SEARCH_PROVIDER: {provider!r} (use 'duckduckgo' or 'tavily')")


DEFAULT_DESCRIPTION = ("Busca en internet información actual o general: noticias, tiempo, datos, "
                       "precios de mercado. Devuelve una lista de resultados (url, título, texto).")


def create_search_tool(name="search_web", description=DEFAULT_DESCRIPTION, max_results=3, region=None):
    """A web search tool for agents. The description is what the model reads."""

    @tool(name, description=description)
    def search_tool(query: str) -> str:
        return json.dumps(search(query, max_results, region=region)["results"], ensure_ascii=False)

    return search_tool


search_web = create_search_tool()
