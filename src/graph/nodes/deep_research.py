"""deep_research 🤖 + review_research ⏸: the CONTEXT each section of the video needs
(research.deep in config.yaml, per format; only when some section of the outline has questions:
otherwise the outline goes straight to the script).

The first research (nodes/research.py) finds the topic; its articles tell the same thing, and a
video is more than that: the problem, how it works, where it comes from, who is who, the risks,
the numbers of reference. A short needs it too: a minute about how something works cannot be made
of a newspaper quote. So, after you approve the outline (each section with the questions it
needs answered, prompts/outline.md):

deep_research: for each section with questions
    1. the LLM writes where to look (prompts/deep_research_searches.md): web searches (Tavily or
       DuckDuckGo, the general web, not only news; never research.skip_domains) and Wikipedia
       articles (services/wikipedia.py, free API, Spanish and English);
    2. up to research.deep_sources_per_section texts are read (paragraphs about the section only)
       and the LLM extracts their facts for that section (prompts/deep_research_facts.md);
    3. the facts join the video's numbered list, with their source and section, and the section
       gets them. Every text read is saved as <run>/research/s<section>_<n>.txt.
review_research: pauses with how many facts each section has and how long they make the video:
    a section can only be as long as its facts allow (lengths.words_by_facts: about
    research.words_per_fact words each); one without facts is left out. Never padded: with less
    material the video is shorter.
    [y] -> write_script (outline.json is read again: remove or change sections there before)
    [n] -> your links: "4 https://... https://... 6 https://..." (section number, then its links):
           they are read for those sections and it asks again.
"""
import re
from typing import Literal
from urllib.parse import urlparse

from langgraph.types import Command

from graph.lengths import assign_words, length_targets, words_by_facts
from graph.script_files import ROLE_LABELS, TYPE_LABELS, load_outline, save_outline
from graph.state import FactsDraft, Outline, SectionSearches
from llmkit.approval import ask_approval
from llmkit.web_search import search
from prompts import load_prompt
from services import wikipedia
from services.articles import blocked, fetch_article
from services.settings import load_settings, runs_dir

NODE = "deep_research"
URL = re.compile(r"https?://\S+")
SHORT_SHARE = 0.6             # a middle section under 60 % of its normal share is reported as short


def _run_id(state, config):
    return state.get("run_id") or config["configurable"]["thread_id"]


def section_header(state, outline, number):
    section = outline.sections[number - 1]
    plan = "\n".join(f"{n}. [{ROLE_LABELS.get(s.role, s.role)}] {s.title}" + ("   <- ESTA SECCIÓN" if n == number else "")
                     for n, s in enumerate(outline.sections, start=1))
    parts = [f"Tema del vídeo: {state['topic']}"]
    if state.get("video_type"):
        parts.append(f"Tipo de vídeo: {TYPE_LABELS[state['video_type']]}")
    parts += [f"Título: {outline.title}", f"Hilo conductor: {outline.angle}",
              "", "# Esquema", plan, "", f"# Sección {number}: {section.title}", section.goal]
    if section.questions:
        parts += ["", "# Lo que hay que averiguar"] + [f"- {q}" for q in section.questions]
    return parts


def searches_request(state, outline, number, facts):
    known = [facts[i - 1]["fact"] for i in outline.sections[number - 1].facts if 1 <= i <= len(facts)]
    return "\n".join(section_header(state, outline, number)
                     + ["", "# Hechos que ya tiene esta sección (no los busques)"] + [f"- {f}" for f in known or ["(ninguno)"]])


def facts_request(state, outline, number, facts, doc, text):
    known = [facts[i - 1]["fact"] for i in outline.sections[number - 1].facts if 1 <= i <= len(facts)]
    return "\n".join(section_header(state, outline, number)
                     + ["", "# Hechos que ya tiene esta sección (no los repitas)"] + [f"- {f}" for f in known or ["(ninguno)"]]
                     + ["", f"# Texto: {doc.get('title') or '(sin título)'}", f"Fuente: {doc['url']}", "", text])


def parse_links(message):
    """'4 https://a https://b 6 https://c' -> {4: [a, b], 6: [c]}; links before any number are ignored."""
    links, section = {}, None
    for token in (message or "").replace(",", " ").split():
        if token.isdigit():
            section = int(token)
        elif URL.match(token) and section:
            links.setdefault(section, []).append(token.rstrip(".;)"))
    return links


def plan_words(outline, targets, settings):
    words_per_fact = settings["research"].get("words_per_fact", 22)
    return words_by_facts([len(s.facts) for s in outline.sections], targets["words"], words_per_fact)


def length_text(seconds):
    """'45 s' for a short, '4.8 minutos' for a long video."""
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} minutos"


def research_report(outline, targets, settings):
    """What each section has and how long the video comes out (for the pause)."""
    words = plan_words(outline, targets, settings)
    normal = assign_words(len(outline.sections), targets["words"])
    lines = []
    for n, (s, w, base) in enumerate(zip(outline.sections, words, normal), start=1):
        note = ("   ⚠️ sin datos: se quita" if w == 0 else
                "   ⚠️ poco: se acorta" if 1 < n < len(words) and w < base * SHORT_SHARE else "")
        lines.append(f"{n}. {s.title}: {len(s.facts)} datos -> ~{w} palabras{note}")
    pace = settings["video"]["words_per_second"]
    lines.append(f"-> da para unos {length_text(sum(words) / pace)} (objetivo: {length_text(targets['seconds'])})")
    return "\n".join(lines)


def make_deep_research(hub, settings=None):
    settings = settings or load_settings()
    plan_searches = hub.chain(load_prompt("deep_research_searches"), schema=SectionSearches, node=NODE)
    extract = hub.chain(load_prompt("deep_research_facts"), schema=FactsDraft, node="facts")   # a cheaper model

    def deep_research(state, config):
        options = settings["research"]
        outline = Outline.model_validate(state["outline"])
        facts, sources = list(state["facts"]), list(state.get("sources", []))
        seen = {s["url"] for s in sources}
        folder = runs_dir(_run_id(state, config)) / "research"
        folder.mkdir(exist_ok=True)
        extra = {int(k): v for k, v in (state.get("extra_urls") or {}).items()}

        # where to look: your links first (always read), then (the first time) the LLM's searches
        todo = {n: [("url", u) for u in urls] for n, urls in extra.items() if 1 <= n <= len(outline.sections)}
        if not state.get("deep_researched"):
            for n, section in enumerate(outline.sections, start=1):
                if not section.questions:
                    continue
                plan = plan_searches.invoke({"text": searches_request(state, outline, n, facts)})
                refs = [("wiki", w) for w in plan.wikipedia]
                for query in plan.web:
                    try:
                        refs += [("url", r["url"]) for r in
                                 search(query, options.get("deep_results_per_query", 3), news=False)["results"]
                                 if not blocked(r["url"], options.get("skip_domains", []))]
                    except Exception as e:                       # one failed search does not stop it
                        print(f"  · deep_research: búsqueda fallida ({query!r}): {e}")
                todo.setdefault(n, []).extend(refs)

        added = {}
        for n, refs in todo.items():
            limit = options.get("deep_sources_per_section", 4) + len(extra.get(n, []))
            wanted = " ".join([outline.sections[n - 1].title, outline.sections[n - 1].goal,
                               *outline.sections[n - 1].questions])
            read = 0
            for kind, ref in refs:
                if read >= limit:
                    break
                try:
                    doc = wikipedia.article(ref) if kind == "wiki" else fetch_article(ref)
                except Exception:
                    doc = None
                if not doc or doc["url"] in seen:
                    continue
                seen.add(doc["url"])
                read += 1
                text = wikipedia.relevant(doc["text"], wanted, options.get("max_article_chars", 6000))
                found = [f.fact.strip() for f in
                         extract.invoke({"text": facts_request(state, outline, n, facts, doc, text)}).facts if f.fact.strip()]
                if not found:
                    continue
                (folder / f"s{n}_{len(sources) + 1:02d}.txt").write_text(f"{doc['url']}\n\n{text}", encoding="utf-8")
                sources.append({"url": doc["url"], "title": doc.get("title", ""), "date": doc.get("date", ""),
                                "outlet": doc.get("outlet") or urlparse(doc["url"]).netloc.removeprefix("www.")})
                first = len(facts) + 1
                facts += [{"fact": f, "source_url": doc["url"], "section": n} for f in found]
                section = outline.sections[n - 1]
                outline.sections[n - 1] = section.model_copy(
                    update={"facts": sorted(set(section.facts) | set(range(first, len(facts) + 1)))})
                added[n] = added.get(n, 0) + len(found)
            print(f"  · deep_research: sección {n} -> {added.get(n, 0)} datos nuevos de {read} textos")

        save_outline(outline, facts, runs_dir(_run_id(state, config)))
        return {"facts": facts, "sources": sources, "outline": outline.model_dump(),
                "deep_researched": True, "extra_urls": {}}

    return deep_research


def make_review_research(settings=None):
    settings = settings or load_settings()

    def review_research(state, config) -> Command[Literal["write_script", "deep_research", "review_research"]]:
        # Nothing with side effects before ask_approval: on resume this node runs again from the start
        folder = runs_dir(_run_id(state, config))
        targets = length_targets(state.get("format") or settings.get("format", "long"), settings)
        outline = Outline.model_validate(state["outline"])
        decision = ask_approval(
            "review_research",
            {"investigación": "\n" + research_report(outline, targets, settings),
             "detalle": f"{folder / 'outline.md'} (cada sección con sus hechos y fuentes)",
             "responde": "[y] escribe el guion con esto (antes puedes quitar o cambiar secciones en "
                         f"{folder / 'outline.json'}) · [n] añade tus fuentes: el número de la sección y sus "
                         "enlaces, p. ej. \"4 https://... https://... 6 https://...\""},
            description="Revisa lo que se ha encontrado para cada sección", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            links = parse_links(decision.get("message"))
            if not links:
                print("  · review_research: no hay enlaces con número de sección (\"4 https://...\"); "
                      "lo pregunto otra vez")
                return Command(goto="review_research")
            return Command(goto="deep_research", update={"extra_urls": links})
        edited = load_outline(folder)                               # your edits to outline.json count
        words = plan_words(edited, targets, settings)
        sections = [s.model_copy(update={"words": w}) for s, w in zip(edited.sections, words) if w > 0]
        dropped = [s.title for s, w in zip(edited.sections, words) if w == 0]
        if dropped:
            print(f"  · review_research: sin datos, fuera del guion: {'; '.join(dropped)}")
        return Command(goto="write_script", update={"outline": edited.model_copy(update={"sections": sections}).model_dump()})

    return review_research
