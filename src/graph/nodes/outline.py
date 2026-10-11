"""outline 🤖 + approve_outline ⏸: the video's sections, approved by you before any script is written.

outline: chosen angle + facts -> title, guiding question and sections (each with its role in the
         story, the emotion it seeks and the facts it uses).
         The words of each section are set by code (lengths.assign_words), not by the LLM.
         Saved as <run>/outline.md (to read) and <run>/outline.json (to edit).
approve_outline: pauses the graph.
         approve -> outline.json is read again (your edits count) -> write_script
                    (long videos with research.deep: -> deep_research first, the context of each
                    section; manual mode: -> manual_script, you write it)
         reject  -> back to outline with your reason as feedback
"""
from typing import Literal

from langgraph.types import Command

from graph.lengths import assign_words, duration_text, length_targets
from graph.script_files import load_outline, outline_markdown, save_outline
from graph.state import Outline, OutlineDraft
from llmkit.approval import ask_approval
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "outline"


def numbered_facts(facts):
    return "\n".join(f"{n}. {f['fact']}" for n, f in enumerate(facts, start=1))


def angle_text(angle):
    """The chosen angle (graph/nodes/angles.py) for a prompt; one written by you may have only a title."""
    lines = [f"Enfoque: {angle['title']}"]
    if angle.get("question"):
        lines.append(f"Pregunta central: {angle['question']}")
    if angle.get("hook"):
        lines.append(f"Gancho propuesto: \"{angle['hook']}\"")
    return "\n".join(lines)


def outline_request(state, targets, feedback=None, previous=None, deep=False):
    parts = [f"Tema del vídeo: {state['topic']}"]
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    parts += [f"Duración: {duration_text(targets['seconds'])} de narración.",
              f"Secciones: entre {targets['min_sections']} y {targets['max_sections']}."]
    if deep:
        parts.append("Vídeo LARGO: construye las secciones con los bloques de las instrucciones y escribe en "
                     "`questions` lo que hay que averiguar para cada una (se investigará después).")
    if state.get("angle"):
        parts += ["", "# Enfoque elegido (el esquema lo desarrolla)", angle_text(state["angle"])]
    parts += ["", "# Hechos disponibles", numbered_facts(state["facts"])]
    if previous:
        parts += ["", "# Esquema anterior", outline_markdown(previous)]
    if feedback:
        parts += ["", "# Corrige esto (rehaz el esquema aplicándolo)", feedback]
    return "\n".join(parts)


def to_outline(draft, n_facts, total_words, max_sections):
    """Draft -> Outline: at most max_sections, valid fact numbers only, words set by code."""
    sections = draft.sections[:max_sections]
    words = assign_words(len(sections), total_words)
    return Outline(title=draft.title, angle=draft.angle, sections=[
        {**s.model_dump(), "facts": sorted({f for f in s.facts if 1 <= f <= n_facts}), "words": w}
        for s, w in zip(sections, words)])


def _run_id(state, config):
    return state.get("run_id") or config["configurable"]["thread_id"]


def deep_research_on(state, settings):
    """research.deep for this run's format (long videos), and not in manual mode (you write it)."""
    video_format = state.get("format") or settings.get("format", "long")
    return bool((settings["research"].get("deep") or {}).get(video_format)) and not state.get("manual")


def make_outline(hub, settings=None):
    settings = settings or load_settings()
    generate = hub.chain(load_prompt(NODE), schema=OutlineDraft, node=NODE)

    def outline(state, config):
        targets = length_targets(state.get("format") or settings.get("format", "long"), settings)
        feedback = state.get("outline_feedback")
        previous = Outline.model_validate(state["outline"]) if feedback and state.get("outline") else None
        deep = deep_research_on(state, settings)
        draft = generate.invoke({"text": outline_request(state, targets, feedback, previous, deep)})
        if len(draft.sections) < targets["min_sections"]:              # one retry with the exact number
            draft = generate.invoke({"text": outline_request(
                state, targets, f"El esquema tiene {len(draft.sections)} secciones y deben ser entre "
                                f"{targets['min_sections']} y {targets['max_sections']}.", draft, deep)})
        result = to_outline(draft, len(state["facts"]), targets["words"], targets["max_sections"])
        save_outline(result, state["facts"], runs_dir(_run_id(state, config)))
        return {"outline": result.model_dump(), "outline_feedback": ""}

    return outline


def make_approve_outline(settings=None):
    settings = settings or load_settings()

    def approve_outline(state, config) -> Command[Literal["outline", "write_script", "manual_script", "deep_research"]]:
        # Nothing with side effects before ask_approval: on resume this node runs again from the start
        folder = runs_dir(_run_id(state, config))
        decision = ask_approval(
            "approve_outline",
            {"esquema": "\n" + outline_markdown(Outline.model_validate(state["outline"]), indent="         "),
             "editar": f"puedes cambiar {folder / 'outline.json'} antes de aprobar"},
            description="Revisa el esquema del vídeo", allowed=("approve", "reject"))
        if decision["type"] == "reject":
            return Command(goto="outline", update={"outline_feedback": decision.get("message") or
                                                   "No me convence: plantéalo de otra forma."})
        edited = load_outline(folder)                                 # your edits to outline.json count
        targets = length_targets(state.get("format") or settings.get("format", "long"), settings)
        words = assign_words(len(edited.sections), targets["words"])  # sections added/removed: rebalance
        edited = edited.model_copy(update={"sections": [s.model_copy(update={"words": w})
                                                        for s, w in zip(edited.sections, words)]})
        goto = ("manual_script" if state.get("manual") else
                "deep_research" if deep_research_on(state, settings) else "write_script")
        return Command(goto=goto, update={"outline": edited.model_dump(), "deep_researched": False})

    return approve_outline
