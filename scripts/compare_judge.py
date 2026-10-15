"""Tests vision models as the judge of visuals on a run's real grids, against answers fixed beforehand.

    python scripts/compare_judge.py prepare --run 20261013-025807 --beats 1 7 10 12 39
    python scripts/compare_judge.py judge --run 20261013-025807 --models google/gemma-4-31b-it anthropic/claude-haiku-5.5 --repeat 3

prepare (free: cached searches, SigLIP on the GPU): rebuilds each beat's grid as the run's judge saw
it (same candidates, SigLIP's best visuals.top_k) -> <run>/prueba_juez/cuadriculas/NN.jpg and NN.json.
Then write <run>/prueba_juez/etiquetas.json by hand, looking at the grids (nobody's model):
    {"10": {"good": [3], "what": {"1": "satélite en una sala blanca", "3": "reloj atómico de laboratorio"}}}
good: the images that show what is said ([] = none: the right answer is to generate one).
judge: every model judges every grid `repeat` times with the pipeline's own prompt and request
(graph/nodes/visuals.py: choice_text) -> <run>/prueba_juez/resultado.md (per beat and per model:
wrong accepts, wrong rejects, what it says it sees) and costes.md.
"""
import argparse
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from graph.build import build_graph
from graph.nodes import visuals
from graph.state import Script, VisualChoice
from llmkit import config
from prompts import load_prompt
from services import LLMHub, contact_sheet, costs, llm, workers
from services.llm_hub import _data_url
from services.settings import load_settings, runs_dir


def run_state(run_id):
    hub = LLMHub(memory="checkpoints")
    values = hub.state(build_graph(hub), run_id)
    if not values or not values.get("beats"):
        raise SystemExit(f"Run {run_id!r} has no beats")
    return values


def prepare(args, folder, values, settings):
    options = settings["visuals"]
    video = settings["video"][values.get("format") or settings.get("format", "long")]
    orientation = "portrait" if video["height"] > video["width"] else "landscape"
    cache_file = runs_dir(args.run) / "visuals" / "searches.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    beats = {b["id"]: visuals.plain(b) for b in values["beats"]}
    wanted = [beats[n] for n in args.beats]
    (folder / "miniaturas").mkdir(parents=True, exist_ok=True)
    found = {b["id"]: visuals.gather(b, orientation, options, folder / "miniaturas", cache) for b in wanted}
    items = [b for b in wanted if found[b["id"]]]
    scores = workers.run("embed", {"model": options.get("embed_model", "google/siglip2-base-patch16-224"),
                                   "items": [{"text": f"{visuals.queries_of(b)[0]}. {b['subject']}",
                                              "images": [c["thumb_path"] for c in found[b["id"]]]}
                                             for b in items]}, settings)["scores"]
    out = folder / "cuadriculas"
    out.mkdir(parents=True, exist_ok=True)
    for b, similarity in zip(items, scores):
        pool = sorted((dict(c, similarity=s) for c, s in zip(found[b["id"]], similarity) if s > -1),
                      key=lambda c: -c["similarity"])[:options.get("top_k", 8)]
        sheet = contact_sheet.grid([open(c["thumb_path"], "rb").read() for c in pool], video["width"] / video["height"])
        (out / f"{b['id']:02d}.jpg").write_bytes(sheet)
        (out / f"{b['id']:02d}.json").write_text(json.dumps(
            {"beat": b, "candidates": [{k: c.get(k) for k in ("provider", "kind", "page_url", "thumb_path", "query")}
                                       for c in pool]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {b['id']:02d}: {len(pool)} imágenes · {b['text'][:70]}")
    for b in wanted:
        if not found[b["id"]]:
            print(f"  {b['id']:02d}: sin candidatos (en el run se generó)")
    print(f"\nCuadrículas en {config.readable_path(out)}. Ahora escribe etiquetas.json mirándolas.")


def verdict(choice, label, min_score):
    """bien | falso_si (accepts an image that does not show it) | falso_no (rejects when a good one exists)
    | dudoso (accepts one of the label's "ok": defensible either way, not counted)."""
    accepted = choice.best > 0 and choice.score >= min_score
    if accepted:
        return "bien" if choice.best in label["good"] else "dudoso" if choice.best in label.get("ok", []) else "falso_si"
    return "falso_no" if label["good"] else "bien"


def captions(folder):
    """fichas.json (each grid's candidates, as their source describes them: NASA's title, file names,
    tags; made by looking them up once) -> {grid: [caption, ...]}, or {} if there is none."""
    path = folder / "fichas.json"
    if not path.exists():
        return {}
    return {key: [re.sub(r"^\d+ \[[^\]]*\] ", "", row) for row in rows]
            for key, rows in json.loads(path.read_text(encoding="utf-8")).items()}


def judge(args, folder, values, settings):
    labels = json.loads((folder / "etiquetas.json").read_text(encoding="utf-8-sig"))
    if args.beats:                                # only these grids (e.g. a quick check)
        labels = {k: v for k, v in labels.items() if int(k) in args.beats}
    fichas = captions(folder) if args.captions else {}
    if args.captions and not fichas:
        raise SystemExit(f"--captions needs {folder / 'fichas.json'}")
    if not args.captions:                         # as the judge was before captions: none at all
        visuals.captions_text = lambda candidates: ""
    script = Script.model_validate(values["script"])
    scenes = {s.id: s for s in script.scenes}
    min_score = settings["visuals"].get("min_score", 5)
    prompt = load_prompt("visual_choice", folder=args.prompt_dir)
    costs.start(folder / args.out)
    results, lines = {}, []
    for model_id in args.models:
        model = llm.model_for("visuals", model=model_id, callbacks=[costs.UsageLog(model_id)])
        ask = model.with_structured_output(VisualChoice, method="function_calling")
        for key in sorted(labels, key=int):
            grid = json.loads((folder / "cuadriculas" / f"{int(key):02d}.json").read_text(encoding="utf-8"))
            beat, pool = grid["beat"], grid["candidates"]
            for c, caption in zip(pool, fichas.get(f"{int(key):02d}", [])):
                c["caption"] = caption
            text = visuals.choice_text(beat, scenes[beat["scene"]], pool, values.get("brief"))
            image = (folder / "cuadriculas" / f"{int(key):02d}.jpg").read_bytes()
            for n in range(args.repeat):
                try:
                    choice = ask.invoke([SystemMessage(prompt), HumanMessage(
                        [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": _data_url(image)}}])])
                    outcome = verdict(choice, labels[key], min_score)
                    row = {"best": choice.best, "score": choice.score, "seen": choice.seen,
                           "reason": choice.reason, "outcome": outcome}
                except Exception as e:
                    row = {"best": 0, "score": 0, "seen": [], "reason": f"ERROR: {str(e)[:200]}", "outcome": "error"}
                results.setdefault(model_id, {}).setdefault(key, []).append(row)
                print(f"  {model_id} · plano {key} · {n + 1}/{args.repeat}: {row['best']} ({row['score']}/10) -> {row['outcome']}")

    lines += ["# Prueba del juez de imágenes", "",
              f"Run {args.run} · nota mínima para usar una imagen: {min_score} · cada modelo, {args.repeat} veces "
              f"por cuadrícula · fichas de las imágenes: {'sí' if args.captions else 'no'} · prompt: "
              f"{args.prompt_dir or 'el actual (src/prompts/visual_choice.md)'}.", "",
              "- **falso sí**: aprueba una imagen que no enseña lo que se dice (el error grave: entra en el vídeo).",
              "- **falso no**: rechaza cuando había una buena (se genera una imagen sin necesidad: cuesta dinero).",
              "- **dudoso**: acepta una imagen defendible (etiqueta `ok`): no cuenta ni como acierto ni como error.", "",
              "| Modelo | Bien | Falso sí | Falso no | Dudoso | Error |", "|---|---|---|---|---|---|"]
    for model_id, per_beat in results.items():
        rows = [r for runs in per_beat.values() for r in runs]
        count = {k: sum(r["outcome"] == k for r in rows) for k in ("bien", "falso_si", "falso_no", "dudoso", "error")}
        lines.append(f"| {model_id} | {count['bien']}/{len(rows)} | {count['falso_si']} | {count['falso_no']} | "
                     f"{count['dudoso']} | {count['error']} |")
    for key in sorted(labels, key=int):
        grid = json.loads((folder / "cuadriculas" / f"{int(key):02d}.json").read_text(encoding="utf-8"))
        good, ok = labels[key]["good"], labels[key].get("ok", [])
        lines += ["", f"## Plano {key}: {grid['beat']['text']}", "",
                  f"Cuadrícula: `cuadriculas/{int(key):02d}.jpg` · buenas: {good or 'ninguna (hay que generar)'}"
                  + (f" · dudosas: {ok}" if ok else "")]
        lines += [f"- {n}: {what}" for n, what in labels[key].get("what", {}).items()]
        if labels[key].get("note"):
            lines.append(f"- *{labels[key]['note']}*")
        for model_id in results:
            for r in results[model_id][key]:
                lines.append(f"- **{model_id}** -> {r['best']} ({r['score']}/10) **{r['outcome']}** · ve: "
                             f"{'; '.join(r['seen'])} · {r['reason']}")
    (folder / f"{args.out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / f"{args.out}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{config.readable_path(folder / f'{args.out}.md')} · {costs.summary(folder / args.out)}")


def main():
    parser = argparse.ArgumentParser(description="Prueba modelos como juez de imágenes con respuestas fijadas antes.")
    parser.add_argument("step", choices=["prepare", "judge"])
    parser.add_argument("--run", required=True, metavar="RUN_ID")
    parser.add_argument("--beats", type=int, nargs="+", help="prepare: the beats to test; judge: only these")
    parser.add_argument("--models", nargs="+", default=["google/gemma-4-31b-it", "anthropic/claude-haiku-5.5"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--captions", action="store_true", help="judge: give the judge each image's caption "
                                                                "(prueba_juez/fichas.json), as the pipeline does now")
    parser.add_argument("--out", default="resultado", help="judge: name of the result files (.md and .json)")
    parser.add_argument("--prompt-dir", help="judge: a folder with another visual_choice.md (default: src/prompts)")
    args = parser.parse_args()
    settings = load_settings()
    folder = runs_dir(args.run) / "prueba_juez"
    values = run_state(args.run)
    if args.step == "prepare":
        if not args.beats:
            parser.error("prepare needs --beats")
        prepare(args, folder, values, settings)
    else:
        judge(args, folder, values, settings)


if __name__ == "__main__":
    main()
