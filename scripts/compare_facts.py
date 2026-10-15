"""Another model extracting the facts of a run's texts, side by side with what the run got (no
graph, nothing in the run changes). To decide if a cheaper model can do research's extraction.

    python scripts/compare_facts.py --run 20261013-025807
    python scripts/compare_facts.py --run 20261013-025807 --model anthropic/claude-haiku-5.5 --limit 5

It reads <run>/research/*.txt (NN.txt: research; sN_NN.txt: deep_research of section N), asks the
model with the same prompt and request each node used, and writes <run>/prueba_hechos/comparacion.md
(the run's facts and the new ones, text by text) and costes.md (what the test cost). The run's own
facts are not asked again: they come from its saved state.
"""
import argparse
import re
import time

from graph.build import build_graph
from graph.nodes import deep_research, research
from graph.state import FactsDraft, Outline
from llmkit import config
from prompts import load_prompt
from services import LLMHub, costs, llm
from services.settings import load_settings, runs_dir

DEEP_FILE = re.compile(r"^s(\d+)_\d+$")


def known_outline(values):
    """The outline as deep_research saw it: each section with only the facts of the first research."""
    outline = Outline.model_validate(values["outline"])
    facts = values["facts"]
    for n, s in enumerate(outline.sections):
        outline.sections[n] = s.model_copy(update={"facts": [i for i in s.facts if "section" not in facts[i - 1]]})
    return outline


def main():
    parser = argparse.ArgumentParser(description="Compara la extracción de hechos de otro modelo con la del run.")
    parser.add_argument("--run", required=True, metavar="RUN_ID")
    parser.add_argument("--model", default="anthropic/claude-haiku-5.5", help="OpenRouter id")
    parser.add_argument("--limit", type=int, help="only the first N texts")
    parser.add_argument("--prompts", metavar="DIR", help="a folder with test versions of research_facts.md and "
                                                         "deep_research_facts.md (default: src/prompts)")
    parser.add_argument("--out", default="prueba_hechos", help="folder inside the run for the results")
    args = parser.parse_args()

    saved = LLMHub(memory="checkpoints")
    values = saved.state(build_graph(saved), args.run)
    if not values or not values.get("facts"):
        raise SystemExit(f"Run {args.run!r} has no research")
    folder = runs_dir(args.run)
    out = folder / args.out
    costs.start(out)
    model = llm.model_for("research", model=args.model, callbacks=[costs.UsageLog(args.model)])
    hub = LLMHub(model=model)
    extract = {"research": hub.chain(load_prompt("research_facts", folder=args.prompts), schema=FactsDraft),
               "deep": hub.chain(load_prompt("deep_research_facts", folder=args.prompts), schema=FactsDraft)}
    max_chars = load_settings()["research"]["max_article_chars"]
    titles = {s["url"]: s for s in values.get("sources", [])}
    outline = known_outline(values) if values.get("outline") else None
    base = [f for f in values["facts"] if "section" not in f]

    files = sorted((folder / "research").glob("*.txt"))[:args.limit]
    lines, totals = [], {"run": 0, "new": 0}
    for n, path in enumerate(files, 1):
        url, _, text = path.read_text(encoding="utf-8").partition("\n\n")
        url = url.strip()
        doc = {"url": url, "title": titles.get(url, {}).get("title", ""), "date": titles.get(url, {}).get("date", ""),
               "text": text}
        deep = DEEP_FILE.match(path.stem)
        if deep:
            section = int(deep.group(1))
            request = deep_research.facts_request(values, outline, section, base, doc, text)
            theirs = [f["fact"] for f in values["facts"] if f["source_url"] == url and f.get("section") == section]
            chain = extract["deep"]
        else:
            request = research.facts_request(values["topic"], values.get("notes", ""), doc, max_chars)
            theirs = [f["fact"] for f in values["facts"] if f["source_url"] == url and "section" not in f]
            chain = extract["research"]
        start = time.time()
        try:
            ours = [f.fact.strip() for f in chain.invoke({"text": request}).facts if f.fact.strip()]
        except Exception as e:
            ours = [f"(ERROR: {str(e)[:200]})"]
        print(f"  {n}/{len(files)} {path.name}: run {len(theirs)} · {args.model} {len(ours)} ({time.time() - start:.0f} s)")
        totals["run"] += len(theirs)
        totals["new"] += len(ours)
        lines += [f"## {path.name} · {doc['title'] or url}", url, "",
                  f"### Run ({len(theirs)})", *[f"- {f}" for f in theirs or ["(ninguno)"]], "",
                  f"### {args.model} ({len(ours)})", *[f"- {f}" for f in ours or ["(ninguno)"]], ""]

    spent = costs.summary(out)
    head = [f"# Hechos: el run frente a {args.model}" + (f" (prompts de {args.prompts})" if args.prompts else ""), "",
            f"Run {args.run} · {values['topic']}", f"{len(files)} textos · hechos del run: {totals['run']} · "
            f"de {args.model}: {totals['new']} · coste de la prueba: {spent or '0 $'}", ""]
    (out / "comparacion.md").write_text("\n".join(head + lines), encoding="utf-8")
    print(f"\n{config.readable_path(out / 'comparacion.md')} · {spent}")


if __name__ == "__main__":
    main()
