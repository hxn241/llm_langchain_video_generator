"""review_script 🤖: an editor reads the whole script before your pause (prompts/review_script.md).

One call with the script, the outline and the facts (and, after a rewrite, what it was asked:
script_requests, to check it was done). It returns:
- issues: what to fix, each with its scene (0 = the whole video) and how. The serious ones (a
  claim that is not in the facts or says more than them, a video that does not answer its topic,
  something key nobody would understand, a hook or a close that does not work) send the script
  back to write_script, at most review.max_revisions times in a row; the rest are only shown.
- dubious: facts the script uses that look wrong (the article may be): never corrected by the
  model, only shown to you, who check them and decide.
What is left goes to <run>/script.md (section "Revisión") and to the choose_script pause, where
[n] with a reason has it rewritten (graph/nodes/manual_script.py).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from langgraph.types import Command

from graph.script_files import ROLE_LABELS, TYPE_LABELS, save_script
from graph.state import Outline, Script, ScriptReview
from llmkit.config import ROOT
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "review_script"
MAX_ISSUES = 8


def review_request(state, video_format, outline, script, facts):
    used = sorted({f for s in outline.sections for f in s.facts}) or range(1, len(facts) + 1)
    roles = {n: ROLE_LABELS.get(s.role, s.role) for n, s in enumerate(outline.sections, start=1)}
    plan = "\n".join(f"{n}. [{roles[n]}] {s.title}: {s.goal}" for n, s in enumerate(outline.sections, start=1))
    lines, section = [], None
    for s in script.scenes:
        if s.section != section:
            section = s.section
            lines.append(f"## Sección {section}" + (f" · {roles[section]}" if section in roles else ""))
        lines.append(f"{s.id}. {s.narration}")
    parts = [f"Tema del vídeo: {state.get('topic') or script.title}",
             f"Formato: {'short vertical (algo más de un minuto)' if video_format == 'short' else 'vídeo largo'}"]
    if state.get("video_type"):
        parts.append(f"Tipo de vídeo: {TYPE_LABELS[state['video_type']]}")
    parts += [f"Título: {outline.title}", f"Hilo conductor: {outline.angle}"]
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    parts += ["", "# Esquema", plan, "", "# Hechos (lo único que el guion puede usar)",
              "\n".join(f"{n}. {facts[n - 1]['fact']}" for n in used if 1 <= n <= len(facts)), "",
              f"# Guion ({script.word_count()} palabras: tus arreglos mantienen la longitud)", *lines]
    if state.get("script_requests"):
        parts += ["", "# Cambios que se pidieron en esta reescritura (comprueba uno a uno que están hechos; "
                      "después, solo lo que la reescritura ha roto o lo grave: \"Si es una reescritura\")",
                  state["script_requests"]]
    return "\n".join(parts)


def log_review(settings, run_id, video_format, revisions, rewrite, report):
    """Every review of every video, one per line (config.yaml: review.log): the errors that come back
    video after video are fixed in the writer's prompt, not by hand in each script."""
    path = (settings.get("review") or {}).get("log")
    if not path:
        return
    path = Path(path) if Path(path).is_absolute() else ROOT / path      # relative: to the project folder
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"date": datetime.now().isoformat(timespec="seconds"), "run_id": str(run_id), "format": video_format,
            "round": revisions, "rewrite": rewrite} | report
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def make_review_script(hub, settings=None):
    settings = settings or load_settings()
    review = hub.chain(load_prompt(NODE), schema=ScriptReview, node=NODE)

    def review_script(state, config) -> Command[Literal["write_script", "choose_script"]]:
        script, outline, facts = Script.model_validate(state["script"]), Outline.model_validate(state["outline"]), state["facts"]
        video_format = state.get("format") or settings.get("format", "long")
        result = review.invoke({"text": review_request(state, video_format, outline, script, facts)})
        ids = {s.id for s in script.scenes}
        issues = [i.model_dump() for i in result.issues if i.scene == 0 or i.scene in ids][:MAX_ISSUES]
        dubious = [d.model_dump() | {"text": facts[d.fact - 1]["fact"], "source_url": facts[d.fact - 1].get("source_url", "")}
                   for d in result.dubious if 1 <= d.fact <= len(facts)]
        report = {"summary": result.summary.strip(), "issues": issues, "dubious": dubious}
        serious = [i for i in issues if i["serious"]]
        revisions, limit = state.get("revisions") or 0, settings.get("review", {}).get("max_revisions", 1)
        log_review(settings, state.get("run_id") or config["configurable"]["thread_id"], video_format,
                   revisions, bool(state.get("script_requests")), report)

        if serious and revisions < limit:
            print(f"  · review_script: {len(serious)} cosas que arreglar -> se reescribe ({revisions + 1}/{limit})")
            for i in serious:
                print(f"      - {'todo el vídeo' if i['scene'] == 0 else 'escena ' + str(i['scene'])}: {i['problem']}")
            return Command(goto="write_script", update={"script_review": report, "script_notes": serious,
                                                        "revisions": revisions + 1})
        save_script(script, runs_dir(state.get("run_id") or config["configurable"]["thread_id"]), report)
        print(f"  · review_script: {len(issues)} notas ({len(serious)} importantes sin resolver) · "
              f"{len(dubious)} hechos dudosos")
        return Command(goto="choose_script", update={"script_review": report, "script_notes": [], "revisions": 0})

    return review_script
