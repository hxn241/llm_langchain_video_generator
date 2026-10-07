"""manual_script ⏸: in manual mode (--manual), YOUR script instead of write_script.

Prepares <run>/manual/ (guion.md with the outline's sections, visuales/ for your files; see
graph/manual.py) and pauses until you say it is ready ([y]). Then what media_escenas.json lists
is downloaded to visuales/ (if there is one) and guion.md is read:
    - problems (empty script, missing files...) -> they are listed and it asks again: fix them
      with the question on screen, or leave with Ctrl+C and go on later with --resume RUN_ID
    - all good -> state["script"] and state["manual_visuals"] (your marks) -> unload_llm -> voice...
[n] asks again too (the run stays paused here).
"""
from typing import Literal

from langgraph.types import Command

from graph import manual
from graph.lengths import length_targets
from graph.script_files import save_script
from graph.state import Outline
from llmkit.approval import ask_approval
from services.settings import load_settings, runs_dir


def make_manual_script(settings=None):
    settings = settings or load_settings()

    def manual_script(state, config) -> Command[Literal["manual_script", "unload_llm"]]:
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        run_folder = runs_dir(run_id)
        targets = length_targets(state.get("format") or settings.get("format", "long"), settings)
        # Before the pause only this (it runs again on resume): it never overwrites your files
        folder = manual.prepare(run_folder, Outline.model_validate(state["outline"]), targets)
        decision = ask_approval(
            "manual_script",
            {"guion": str(folder / manual.SCRIPT_FILE), "imágenes y vídeos": str(folder / manual.MEDIA_FOLDER),
             "o una página de Pexels por escena": str(folder / manual.MEDIA_LIST),
             "esquema y hechos": str(run_folder / "outline.md"),
             "cuando esté": "[y] sigue con la voz y el montaje · Ctrl+C para salir y seguir otro día con "
                            f"python scripts/run_pipeline.py --resume {run_id}"},
            description="Escribe tu guion y deja tus imágenes", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            return Command(goto="manual_script")
        problems = manual.fetch(run_folder)                      # media_escenas.json -> visuales/
        script, marks, more = manual.parse(run_folder, [s["url"] for s in state.get("sources", [])])
        problems += [p for p in more if p not in problems]
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
