"""choose_script ⏸ and manual_script ⏸: the script write_script made, or YOURS.

choose_script: after write_script it asks. [y] goes on with that script; [n] -> manual_script
(and the run becomes a manual one: state["manual"], --redo reads guion.md).

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
from graph.script_files import save_script
from graph.state import Outline, Script
from llmkit.approval import ask_approval
from services.settings import load_settings, runs_dir


def make_choose_script():
    def choose_script(state, config) -> Command[Literal["unload_llm", "manual_script"]]:
        # Nothing with side effects before ask_approval: on resume this node runs again from the start
        run_folder = runs_dir(state.get("run_id") or config["configurable"]["thread_id"])
        script = Script.model_validate(state["script"])
        save_script(script, run_folder)                         # script.md up to date to read it
        decision = ask_approval(
            "choose_script",
            {"guion": f"{script.title!r} · {len(script.scenes)} escenas · {script.word_count()} palabras",
             "léelo en": str(run_folder / "script.md"),
             **({"avisos": "; ".join(state["script_warnings"])} if state.get("script_warnings") else {}),
             "responde": "[y] sigo con este guion · [n] escribo el mío (guion.md ya lo trae para editarlo; "
                         "Enter en el motivo)"},
            description="¿Usas este guion o escribes el tuyo?", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            return Command(goto="manual_script", update={"manual": True})
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
