"""choose_script ⏸ and manual_script ⏸: the script write_script made, or YOURS.

choose_script: after write_script and review_script it asks, showing what the review left (notes,
dubious facts; all in script.md). [y] goes on with that script; [n] with a reason -> write_script
rewrites it applying your reason (and it is reviewed again); [n] without a reason (Enter) ->
manual_script (and the run becomes a manual one: state["manual"], --redo reads guion.md).

manual_script: prepares <run>/manual/ (guion.md with the outline's sections and, if write_script
ran, its scenes to keep, edit or replace; visuales/ for your files; see graph/manual.py) and
pauses until you say it is ready ([y]). Also reached straight from the outline with --manual
(then guion.md has only the sections). Then guion.md is read:
    - problems (empty script, missing files...) -> they are listed and it asks again: fix them
      with the question on screen, or leave with Ctrl+C and go on later with --resume RUN_ID
    - all good -> state["script"] and state["manual_visuals"] (your marks) -> unload_llm -> voice...
[n] goes back to write_script's script (if there is one; with --manual it asks again).
"""
from typing import Literal

from langgraph.types import Command

from graph import manual
from graph.lengths import length_targets
from graph.script_files import save_script, where
from graph.state import Outline, Script
from llmkit.approval import ask_approval
from services.settings import load_settings, runs_dir

INDENT = "         "
# A reject without a reason: what llmkit.console sends on Enter, and reject_all (evaluations)
NO_REASON = {"", "Rechazado por el usuario.", "Rechazado automáticamente (evaluación)."}


def review_lines(review):
    """What review_script left, for the pause (the whole review is in script.md)."""
    lines = [review.get("summary") or "Sin problemas."]
    lines += [f"⚠️ hecho {d['fact']} dudoso: \"{d['text'][:90]}\" — {d['why']}" for d in review.get("dubious", [])]
    lines += [f"· {where(i['scene'])}{' (importante)' if i['serious'] else ''}: {i['problem']}"
              for i in review.get("issues", [])]
    return "\n" + "\n".join(INDENT + line for line in lines)


def make_choose_script():
    def choose_script(state, config) -> Command[Literal["unload_llm", "manual_script", "write_script"]]:
        # Nothing with side effects before ask_approval: on resume this node runs again from the start
        run_folder = runs_dir(state.get("run_id") or config["configurable"]["thread_id"])
        script = Script.model_validate(state["script"])
        review = state.get("script_review")
        save_script(script, run_folder, review)                 # script.md up to date to read it
        decision = ask_approval(
            "choose_script",
            {"guion": f"{script.title!r} · {len(script.scenes)} escenas · {script.word_count()} palabras",
             "léelo en": str(run_folder / "script.md"),
             **({"revisión": review_lines(review)} if review else {}),
             **({"avisos": "; ".join(state["script_warnings"])} if state.get("script_warnings") else {}),
             "responde": "[y] sigo con este guion · [n] y escribe qué cambiar: lo reescribe (p. ej. \"no uses "
                         "el hecho 5\", \"aplica las notas\") · [n] y Enter: escribo el mío (guion.md ya lo "
                         "trae para editarlo)"},
            description="¿Usas este guion, lo reescribe o escribes el tuyo?", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            reason = (decision.get("message") or "").strip()
            if reason in NO_REASON:
                return Command(goto="manual_script", update={"manual": True})
            return Command(goto="write_script", update={"review_feedback": reason, "script_notes": [],
                                                        "revisions": 0})
        return Command(goto="unload_llm")

    return choose_script


def make_manual_script(settings=None):
    settings = settings or load_settings()

    def manual_script(state, config) -> Command[Literal["manual_script", "unload_llm"]]:
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        run_folder = runs_dir(run_id)
        targets = length_targets(state.get("format") or settings.get("format", "long"), settings)
        # Before the pause only this (it runs again on resume): it never overwrites your files
        written = Script.model_validate(state["script"]) if state.get("script") else None
        folder = manual.prepare(run_folder, Outline.model_validate(state["outline"]), targets, written)
        decision = ask_approval(
            "manual_script",
            {"guion": str(folder / manual.SCRIPT_FILE), "imágenes y vídeos": str(folder / manual.MEDIA_FOLDER),
             "esquema y hechos": str(run_folder / "outline.md"),
             "responde": "[y] listo: sigue con la voz · "
                         + ("[n] mejor el guion automático · " if written else "")
                         + f"Ctrl+C para salir y seguir otro día con python scripts/run_pipeline.py --resume {run_id}"},
            description="Escribe tu guion", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            if written:
                return Command(goto="unload_llm", update={"manual": False})
            return Command(goto="manual_script")
        script, marks, problems = manual.parse(run_folder, [s["url"] for s in state.get("sources", [])])
        if problems:
            print("\n  ⚠️  El guion tiene problemas (corrígelos y responde [y] otra vez):\n"
                  + "\n".join(f"       - {p}" for p in problems))
            return Command(goto="manual_script")
        save_script(script, run_folder)
        print(f"  · manual_script: {len(script.scenes)} escenas · {script.word_count()} palabras · "
              f"{sum(map(len, marks.values()))} marcas tuyas en {len(marks)} escenas "
              f"({len(script.scenes) - len(marks)} escenas con imágenes automáticas)")
        return Command(goto="unload_llm", update={"script": script.model_dump(), "manual_visuals": marks,
                                                   "script_warnings": []})

    return manual_script
