"""Launches or resumes a video run.

    python scripts/run_pipeline.py                              # the topic, chosen from the feeds' ones
    python scripts/run_pipeline.py --format short               # ... and a short whatever the topic
    python scripts/run_pipeline.py --topic "La subida del precio del alquiler en España"
    python scripts/run_pipeline.py --topic "..." --notes "enfócalo en los jóvenes" --url https://...
    python scripts/run_pipeline.py --topic "..." --format short
    python scripts/run_pipeline.py --topic "..." --type explainer    # news | explainer (default: the model decides)
    python scripts/run_pipeline.py --resume 20261004-153000      # a run left paused (e.g. at the outline)
    python scripts/run_pipeline.py --resume 20261004-153000 --redo visuals   # redo a media node and what follows
    python scripts/run_pipeline.py --resume 20261004-153000 --redo voice --keep-visuals   # only a new voice:
                                                     # the same images on its new times, no paid calls

--type: news (something that has just happened: news searches) or explainer (something that does
not expire, like how a thing works: the general web + Wikipedia). Without it the model decides
from the topic and says what it chose.

Without --topic it starts with the topics of the feeds (config.yaml: topics; src/services/topics.py):
the new entries scored by the model plus the ones proposed before and not made yet. It stops to choose:
[y] the first, [e] another number, [n] + "x 3 5" discards those for good, [n] + Enter shows others.
The chosen one brings its articles (--url), its datazo (--notes), its type and its format (unless you
gave --type or --format). Every run's topic is recorded in data/temas/registro.json (en_curso; subido
after upload_youtube.py), so it is not proposed again.

It stops four times:
  1. the angle: 3 proposals; [y] the recommended one, [e] another number or your own angle as text,
     [n] three new ones (with your reason). angles.json can be edited before approving.
  2. the outline: you read it (and may edit outline.json), then approve it or reject it with a reason.
     If its sections have questions (research.deep), each section's context is looked up
     (deep_research: web + Wikipedia) and it stops once more (review_research): facts per section
     and how long the video comes out; [y] go on, [n] add your links ("4 https://... 6 https://...":
     section number, then its links).
  3. the script (script.md, with the review of an editor model: notes and dubious facts): [y] go on
     with it; [n] + what to change: it is rewritten (and reviewed again); [n] + Enter: write yours:
     <run>/manual/guion.md comes with its scenes to edit or replace, with [file.jpg] marks where
     your images/videos go (visuales/). See src/graph/manual.py.
  4. before assembling: <run>/montaje/ has everything in order (01.mp4, 02.jpg...; lista.md says what
     each one is). Replace, remove or add files there and [y]: it is assembled exactly as it is,
     without any judge. [n] stops; `--redo assemble` reads the folder again (see src/graph/montage.py).
     manual/prompts_imagenes.md has every beat's prompt, to make a replacement with your own AI.
--yes approves all of them on its own (tests): the written script, the visuals as chosen.

    python scripts/run_pipeline.py --topic "..." --manual       # straight to your script (no write_script)

In a normal run, --redo reads script.json again: changed narration -> --redo voice (the images keep
their plan, beats.json, and follow the new words; nothing new is generated).
In a manual run (yours at pause 3), --redo reads guion.md again: replaced files -> --redo visuals; changed marks
-> --redo beats; changed narration -> --redo voice.
--redo (voice | align | brief | beats | labels | visuals | assemble) runs that node again and the ones
after it, also on a finished run (e.g. one made before that node existed). brief.json, beats.json and
rotulos.json are read again if they exist (your edits count): delete them for new ones. visuals/choices.json
too: the stock already picked is kept and only the rest is generated (delete it to judge everything
again; the generated images are reused while their prompt does not change). A brief.json
newer than beats.json (new or edited) makes new beats. Only the labels (rótulos) again, keeping the
images: delete rotulos.json and --redo labels --keep-visuals. Only edited their texts: --redo assemble.

Each run is a LangGraph thread (id = run_id) saved in <STATE_DIR>/checkpoints.sqlite,
and its files go to <STATE_DIR>/runs/<run_id>/.
"""
import argparse
from datetime import datetime

from graph import manual
from graph.build import REDO_AFTER, build_graph
from graph.script_files import load_script, save_script
from graph.state import Script
from llmkit import config
from services import LLMHub, approve_all, ask_decisions, costs, run_cli, topics
from services.settings import load_settings, runs_dir

REDO_NEEDS = {"voice": ["script"], "align": ["script", "audio"], "brief": ["script", "audio"],
              "beats": ["script", "audio"], "labels": ["script", "audio"],
              "visuals": ["script", "audio", "beats"], "assemble": ["script", "audio", "visuals"]}


def parse_args():
    parser = argparse.ArgumentParser(description="Genera (o reanuda) un vídeo sobre un tema.")
    parser.add_argument("--topic", help="what the video is about (without it: chosen from the feeds' topics)")
    parser.add_argument("--notes", default="", help="your angle, opinion or what to stress")
    parser.add_argument("--url", action="append", default=[], help="an article to use (repeatable)")
    parser.add_argument("--format", choices=["short", "long"], help="default: `format` in config.yaml")
    parser.add_argument("--type", choices=["news", "explainer"], help="news: something that has just happened; "
                                                                      "explainer: something that does not expire "
                                                                      "(default: the model decides)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a paused run")
    parser.add_argument("--redo", choices=sorted(REDO_AFTER), help="with --resume: run this node again "
                                                                       "and the ones after it")
    parser.add_argument("--keep-visuals", action="store_true",
                        help="with --redo voice/align/brief/beats: the same images on the new times (nothing "
                             "searched, judged or generated: no paid calls)")
    parser.add_argument("--yes", action="store_true", help="approve every pause without asking (the written script, "
                                                           "no images of yours)")
    parser.add_argument("--manual", action="store_true", help="straight to your script, without write_script "
                                                              "(after the outline it pauses: see <run>/manual/)")
    args = parser.parse_args()
    if (args.topic or args.url or args.notes) and args.resume:
        parser.error("--topic, --url and --notes are chosen when the run starts")
    if args.manual and args.resume:
        parser.error("--manual is chosen when the run starts (a resumed run keeps its mode)")
    if args.type and args.resume:
        parser.error("--type is chosen when the run starts (a resumed run keeps its research)")
    if args.redo and not args.resume:
        parser.error("--redo needs --resume RUN_ID")
    if args.keep_visuals and args.redo not in ("voice", "align", "brief", "beats", "labels"):
        parser.error("--keep-visuals goes with --redo voice | align | brief | beats | labels")
    return args


def manual_update(values, run_id, redo):
    """Manual run + --redo: guion.md is read again (your changes count). Changed narration needs a
    new voice (--redo voice); changed marks need new beats (--redo beats or earlier)."""
    script, marks, problems = manual.parse(runs_dir(run_id), [s["url"] for s in values.get("sources", [])])
    if problems:
        raise SystemExit("El guion tiene problemas:\n" + "\n".join(f"  - {p}" for p in problems))
    order = list(REDO_AFTER)                                    # voice, align, beats, visuals, assemble
    if ([s.narration for s in script.scenes] != [s["narration"] for s in values["script"]["scenes"]]
            and order.index(redo) > order.index("voice")):
        raise SystemExit("Has cambiado la narración de guion.md: la voz ya no coincide. Usa --redo voice")
    old_marks = {int(k): v for k, v in (values.get("manual_visuals") or {}).items()}
    if marks != old_marks and order.index(redo) > order.index("beats"):
        raise SystemExit("Has cambiado las marcas [archivo] de guion.md: usa --redo beats")
    save_script(script, runs_dir(run_id))
    return {"script": script.model_dump(), "manual_visuals": marks}


def script_update(values, run_id, redo):
    """Normal run + --redo: script.json is read again (your changes count). Changed narration
    needs a new voice (--redo voice): the images keep their plan and follow the new words."""
    folder = runs_dir(run_id)
    if not (folder / "script.json").exists():
        return {}
    script = load_script(folder)
    if script.model_dump() == Script.model_validate(values["script"]).model_dump():
        return {}
    order = list(REDO_AFTER)
    if ([s.narration for s in script.scenes] != [s["narration"] for s in values["script"]["scenes"]]
            and order.index(redo) > order.index("voice")):
        raise SystemExit("Has cambiado la narración de script.json: la voz ya no coincide. Usa --redo voice")
    save_script(script, folder)                                 # script.md as well
    print(f"  · script.json leído de nuevo: {len(script.scenes)} escenas · {script.word_count()} palabras")
    return {"script": script.model_dump()}


def run_graph(hub, app, inputs, run_id, approve):
    """hub.run, with every paid call recorded in <run>/costes.jsonl; costes.md is written at the end,
    also when the run stops on an error (e.g. no credit) or at a pause."""
    costs.start(runs_dir(run_id))
    try:
        return hub.run(app, inputs, thread=run_id, approve=approve, verbose=True)
    finally:
        spent = costs.summary(runs_dir(run_id))
        if spent:
            print(f"\nCoste del run hasta ahora: {spent} ({config.readable_path(runs_dir(run_id) / costs.SUMMARY)})")


def main():
    args = parse_args()
    hub = LLMHub(memory="checkpoints")
    app = build_graph(hub)
    approve = approve_all if args.yes else ask_decisions

    if args.resume:
        run_id = args.resume
        thread = {"configurable": {"thread_id": run_id}}
        values = hub.state(app, run_id)
        if not values:
            raise SystemExit(f"Run {run_id!r} not found")
        if args.redo:
            missing = [k for k in REDO_NEEDS[args.redo] if not values.get(k)]
            if missing:
                raise SystemExit(f"Run {run_id!r} has no {', '.join(missing)} yet: it cannot redo {args.redo}")
            update = (manual_update if values.get("manual") else script_update)(values, run_id, args.redo)
            if args.keep_visuals:
                if not values.get("visuals"):
                    raise SystemExit(f"Run {run_id!r} has no images yet: --keep-visuals has nothing to keep")
                update = {**update, "keep_visuals": True}
            app.update_state(thread, update, as_node=REDO_AFTER[args.redo])   # the graph goes on from there
        elif not app.get_state(thread).next:
            raise SystemExit(f"Run {run_id!r} already finished: nothing to resume (use --redo NODE)")
        state = run_graph(hub, app, None, run_id, approve)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Without --topic, the chosen topic brings its format (unless --format)
        video_format = args.format or (load_settings().get("format", "long") if args.topic else None)
        if args.topic:
            print(f"Run {run_id} · {video_format} · {args.topic}")
            topics.start_run(run_id, topic=args.topic, urls=args.url)
        else:
            print(f"Run {run_id} · {video_format or 'formato del tema'} · buscando temas")
        state = run_graph(hub, app, {"run_id": run_id, "notes": args.notes, "urls": args.url, "manual": args.manual,
                                     **({"topic": args.topic} if args.topic else {}),
                                     **({"format": video_format} if video_format else {}),
                                     **({"video_type": args.type} if args.type else {})},
                          run_id, approve)

    folder = runs_dir(run_id)
    if state.get("sources"):
        print(f"\nInvestigación: {len(state['sources'])} artículos · {len(state.get('facts', []))} hechos")
    if state.get("script"):
        script = Script.model_validate(state["script"])
        json_path, md_path = save_script(script, folder, None if state.get("manual") else state.get("script_review"))
        print(f"\nGuion: {script.title!r} · {len(script.scenes)} escenas · {script.word_count()} palabras "
              f"(~{script.word_count() / load_settings()['video']['words_per_second'] / 60:.1f} min)")
        print(f"  {config.readable_path(md_path)}\n  {config.readable_path(json_path)}")
    for warning in state.get("script_warnings", []):
        print(f"  ⚠️  {warning}")
    if state.get("audio"):
        total = sum(a["duration"] for a in state["audio"].values())
        print(f"\nVoz: {len(state['audio'])} audios · {total / 60:.1f} min en total")
        print(f"  {config.readable_path(folder / 'audio')}")
    if state.get("beats"):
        print(f"\nFragmentos visuales: {len(state['beats'])}")
        print(f"  {config.readable_path(folder / 'beats.json')}")
    if state.get("visuals"):
        kinds = [v["kind"] for v in state["visuals"].values()]
        counts = [f"{kinds.count(k)} {label}" for k, label in
                  [("video", "vídeos encontrados"), ("photo", "fotos encontradas"), ("generated", "generadas"),
                   ("none", "sin visual")]
                  if k in kinds]
        print(f"\nVisuales: {' · '.join(counts)}")
        print(f"  {config.readable_path(folder / 'visuals')} (choices.json: qué se eligió y por qué)")
        if any(v.get("ai_generated") for v in state["visuals"].values()):
            print("  ⚠️  hay imágenes generadas con IA: márcalo como contenido sintético al subirlo")
    for warning in state.get("visual_warnings", []):
        print(f"  ⚠️  {warning}")
    if state.get("video_path"):
        print(f"\nVídeo: {config.readable_path(state['video_path'])}")


if __name__ == "__main__":
    run_cli(main)
