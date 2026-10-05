"""angles 🤖 + choose_angle ⏸: from where the story is told, chosen by you before the outline.

angles:       facts -> 3 different angles (question, hook, why it hooks), and the strongest one.
              The obvious "news summary" angle is what makes a flat video: the prompt looks for
              the conflict, the surprise or what matters to the viewer.
              Saved as <run>/angles.md (to read) and <run>/angles.json (to edit).
choose_angle: pauses the graph.
              approve -> the recommended one (angles.json is read again: your edits count)
              edit    -> "elegido": another number, or your own angle written as text
              reject  -> 3 new angles, with your reason as feedback
"""
from typing import Literal

from langgraph.types import Command

from graph.lengths import duration_text, length_targets
from graph.nodes.outline import numbered_facts
from graph.script_files import angles_markdown, load_angles, save_angles
from graph.state import Angle, AnglesDraft
from llmkit.approval import ask_approval
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "angles"
HOW_TO_CHOOSE = ("[y] el recomendado · [e] cambia 'elegido' por otro número o escribe tu propio enfoque · "
                 "[n] otros tres (di qué no te convence)")


def angles_request(state, seconds, feedback=None, previous=None):
    parts = [f"Tema del vídeo: {state['topic']}"]
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    parts += [f"Duración: {duration_text(seconds)} de narración.", "", "# Hechos disponibles",
              numbered_facts(state["facts"])]
    if previous:
        parts += ["", "# Enfoques anteriores (no los repitas)", angles_markdown(previous)]
    if feedback:
        parts += ["", "# Lo que no convenció (proponlos de nuevo teniéndolo en cuenta)", feedback]
    return "\n".join(parts)


def _run_id(state, config):
    return state.get("run_id") or config["configurable"]["thread_id"]


def make_angles(hub, settings=None):
    settings = settings or load_settings()
    generate = hub.chain(load_prompt(NODE), schema=AnglesDraft, node=NODE)

    def angles(state, config):
        targets = length_targets(state.get("format") or settings.get("format", "long"), settings)
        feedback = state.get("angle_feedback")
        previous = AnglesDraft.model_validate(state["angles"]) if feedback and state.get("angles") else None
        draft = generate.invoke({"text": angles_request(state, targets["seconds"], feedback, previous)})
        n_facts = len(state["facts"])
        draft = draft.model_copy(update={"angles": [a.model_copy(update={"facts": sorted({f for f in a.facts
                                                                                          if 1 <= f <= n_facts})})
                                                    for a in draft.angles]})
        save_angles(draft, runs_dir(_run_id(state, config)))
        return {"angles": draft.model_dump(), "angle_feedback": ""}

    return angles


def make_choose_angle():
    def choose_angle(state, config) -> Command[Literal["angles", "outline"]]:
        # Nothing with side effects before ask_approval: on resume this node runs again from the start
        folder = runs_dir(_run_id(state, config))
        proposed = AnglesDraft.model_validate(state["angles"])
        decision = ask_approval(
            "choose_angle",
            {"enfoques": "\n" + angles_markdown(proposed, indent="         ") + "         " + HOW_TO_CHOOSE
                         + f"\n         (también puedes cambiar {folder / 'angles.json'} antes de aprobar)",
             "elegido": proposed.angles.index(proposed.recommended()) + 1},
            description="Elige el enfoque del vídeo", allowed=("approve", "edit", "reject"))
        if decision["type"] == "reject":
            return Command(goto="angles", update={"angle_feedback": decision.get("message") or
                                                  "Ninguno me convence: busca enfoques más originales."})
        angles = load_angles(folder)                                  # your edits to angles.json count
        chosen = angles.recommended()
        if decision["type"] == "edit":
            picked = decision["edited_action"]["args"].get("elegido")
            if isinstance(picked, int) and 1 <= picked <= len(angles.angles):
                chosen = angles.angles[picked - 1]
            elif isinstance(picked, str) and picked.strip():            # your own angle
                chosen = Angle(title=picked.strip(), question="", why="Propuesto por el autor", hook="")
            else:
                print(f"  · choose_angle: {picked!r} is not 1-{len(angles.angles)} nor text: "
                      f"using the recommended one")
        return Command(goto="outline", update={"angle": chosen.model_dump()})

    return choose_angle
