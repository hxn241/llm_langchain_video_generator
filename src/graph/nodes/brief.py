"""brief 🤖: the visual universe of the whole video -> state["brief"], once per video.

One call with the final script (yours or write_script's): the filmable objects and places that
give the video its identity (universe), what must never be seen (avoid), how each figurative
expression is shown (metaphors: "the copper wall is falling" is not a demolition) and the
image style (replaces visuals.image_style). beats plans every shot inside it, prompts_imagenes.md
starts with it and the vision judge gets its `avoid` (prompts/brief.md).

Saved to <run>/brief.json: edit it and `--redo brief` reads it again (and makes new beats);
delete it for a new brief.
"""
import json

from graph.state import BriefDraft, Script
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "brief"


def brief_request(state, script):
    parts = [f"Tema del vídeo: {state.get('topic') or script.title}", f"Título: {script.title}"]
    if state.get("angle", {}).get("title"):
        parts.append(f"Enfoque: {state['angle']['title']}")
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    parts += ["", "# Guion completo"] + [f"{s.id}. {s.narration}" for s in script.scenes]
    return "\n".join(parts)


def brief_lines(brief, indent=""):
    """The brief as readable lines (for the beats request, prompts_imagenes.md and the judge)."""
    if not brief:
        return []
    lines = [f"{indent}Universo visual: {', '.join(brief.get('universe', []))}",
             f"{indent}Nunca debe verse: {', '.join(brief.get('avoid', [])) or '-'}",
             f"{indent}Estilo: {brief.get('style', '')}"]
    for m in brief.get("metaphors", []):
        lines.append(f"{indent}Metáfora \"{m['phrase']}\" -> se ve: {m['show']}")
    return lines


def make_brief(hub, settings=None):
    settings = settings or load_settings()
    write = hub.chain(load_prompt(NODE), schema=BriefDraft, node=NODE)

    def brief(state, config):
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        brief_file = runs_dir(run_id) / "brief.json"
        if brief_file.exists():
            found = BriefDraft.model_validate_json(brief_file.read_text(encoding="utf-8")).model_dump()
            print(f"  · brief: de {brief_file.name} (bórralo para hacer uno nuevo)")
            return {"brief": found}
        script = Script.model_validate(state["script"])
        draft = write.invoke({"text": brief_request(state, script)})
        draft.style = draft.style.strip() or settings["visuals"].get("image_style", "")
        brief_file.write_text(json.dumps(draft.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  · brief: {len(draft.universe)} objetos del universo visual · {len(draft.avoid)} prohibidos · "
              f"{len(draft.metaphors)} metáforas")
        return {"brief": draft.model_dump()}

    return brief
