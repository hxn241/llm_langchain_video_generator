"""review_visuals ⏸: after visuals, before assemble. You check <run>/montaje/ (graph/montage.py):
one file per beat in order (01.mp4, 02.jpg...) and lista.md (time, text and origin of each).
Replace, remove or add files there, then:
[y] -> assemble uses exactly what is in the folder, without any judge.
[n] -> stops without assembling; later `--redo assemble` reads the folder again and comes back here.
"""
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command

from graph import montage
from llmkit.approval import ask_approval
from services.settings import runs_dir


def make_review_visuals():
    def review_visuals(state, config) -> Command[Literal["assemble", "__end__"]]:
        run_folder = runs_dir(state.get("run_id") or config["configurable"]["thread_id"])
        folder = montage.folder_of(run_folder)
        if not folder.exists():               # a run from before montaje/ existed. Only when missing: on
            montage.write(run_folder, state["beats"], state.get("visuals") or {},   # resume it must not undo yours
                          state.get("visual_extras"))
        decision = ask_approval(
            "review_visuals",
            {"carpeta": str(folder), "lista": str(folder / montage.LIST_FILE),
             "responde": "[y] monta con lo que hay en la carpeta (sin juez) · [n] para sin montar "
                         "(luego: --redo assemble)"},
            description="Revisa lo que va al vídeo", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            print("  · review_visuals: sin montar. Cuando esté lista la carpeta: --redo assemble")
            return Command(goto=END)
        found, problems = montage.read(run_folder, state["beats"])
        for p in problems:
            print(f"  ⚠️  {p}")
        removed = [n for n, files in found.items() if not files]
        added = [n for n, files in found.items() if len(files) > 1]
        print(f"  · review_visuals: {sum(map(len, found.values()))} archivos"
              + (f" · sin archivo (sigue el anterior): {', '.join(map(str, sorted(removed)))}" if removed else "")
              + (f" · con varios: {', '.join(map(str, sorted(added)))}" if added else ""))
        return Command(goto="assemble", update={"montage": found})

    return review_visuals
