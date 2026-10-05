"""write_script 🤖: approved outline (+ chosen angle) + facts -> Script, ONE SECTION PER CALL.

A 9B model cannot write 10 minutes (~1,600 words) well in one go: it shortens,
repeats itself or loses the thread. Each section is written on its own, seeing
the whole outline, its own facts and how the previous section ended.

- Each section's words come from the outline (lengths.assign_words) and its scenes
  from the format. If a section misses them, it is asked ONCE more with the numbers.
- With review_feedback (from the review node) each section is rewritten from its
  previous version applying it.
- The sources are the articles of the research: the LLM never writes them.
"""
import json

from graph.lengths import length_problems, length_targets, section_targets
from graph.script_files import ROLE_LABELS
from graph.state import Outline, Script, SectionDraft
from prompts import load_prompt
from services.settings import load_settings

NODE = "write_script"
FORMAT_LABELS = {"short": "Short vertical (9:16)", "long": "vídeo horizontal (16:9)"}
CONTEXT_SCENES = 2            # scenes of the previous section shown for continuity


def section_request(state, outline, number, targets, facts, video_format, before=None,
                    previous=None, feedback=None):
    section = outline.sections[number - 1]
    total = len(outline.sections)
    plan = "\n".join(f"{n}. [{ROLE_LABELS[s.role]}] {s.title}: {s.goal}"
                     + ("   <- ESTA SECCIÓN" if n == number else "")
                     for n, s in enumerate(outline.sections, start=1))
    position = ("Es la PRIMERA sección: la primera escena es el gancho." if number == 1 else
                "Es la ÚLTIMA sección: cierra con la conclusión y una reflexión o pregunta." if number == total else
                "Es una sección intermedia.")
    used = section.facts or range(1, len(facts) + 1)                 # no facts listed: all of them
    angle = state.get("angle") or {}
    parts = [f"Formato: {FORMAT_LABELS[video_format]}.",
             f"Título del vídeo: {outline.title}", f"Hilo conductor: {outline.angle}"]
    if angle.get("title"):
        parts.append(f"Enfoque: {angle['title']}")
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    parts += ["", "# Esquema", plan, "",
              f"# Escribe la sección {number} de {total}: {section.title}", section.goal,
              f"Función en la historia: {ROLE_LABELS[section.role]}. Emoción que busca: {section.emotion}.",
              position]
    if number == 1 and angle.get("hook"):
        parts.append(f"Gancho propuesto (úsalo o mejóralo, sin salirte de los hechos): \"{angle['hook']}\"")
    parts += [f"Longitud: entre {targets['min_scenes']} y {targets['max_scenes']} escenas, y entre "
              f"{targets['min_words']} y {targets['max_words']} palabras de narración (objetivo: {targets['words']}).",
              "", "# Hechos para esta sección",
              "\n".join(f"{n}. {facts[n - 1]['fact']}" for n in used)]
    if before:
        parts += ["", "# Así termina la sección anterior (enlaza con ella, no la repitas)",
                  "\n".join(s.narration for s in before)]
    if previous:
        parts += ["", "# Versión anterior de esta sección",
                  json.dumps([s.model_dump(exclude={"id", "section"}) for s in previous], ensure_ascii=False, indent=1)]
    if feedback:
        parts += ["", "# Corrige esto (reescribe la sección aplicándolo)", feedback]
    return "\n".join(parts)


def make_write_script(hub, settings=None):
    """The node, bound to an LLMHub (its model for this node comes from config.yaml: llm.write_script)."""
    settings = settings or load_settings()
    generate = hub.chain(load_prompt(NODE), schema=SectionDraft, node=NODE)

    def write_script(state):
        outline, facts = Outline.model_validate(state["outline"]), state["facts"]
        video_format = state.get("format") or settings.get("format", "long")
        targets = length_targets(video_format, settings)
        # Rewriting after a review: each section starts from its previous version and the feedback
        old = Script.model_validate(state["script"]) if state.get("script") else None
        feedback = state.get("review_feedback") if old else None

        sections, warnings = [], []
        for number, section in enumerate(outline.sections, start=1):
            goal = section_targets(section.words, targets["words_per_scene"])
            before = sections[-1][-CONTEXT_SCENES:] if sections else None
            previous = old.section_scenes(number) if old else None

            def write(previous, feedback):
                return generate.invoke({"text": section_request(state, outline, number, goal, facts, video_format,
                                                                before, previous, feedback)}).scenes

            scenes = write(previous, feedback)
            where = f"La sección {number}"
            problems = length_problems(sum(len(s.narration.split()) for s in scenes), len(scenes), goal, where)
            if problems:                                               # one retry with the exact numbers
                scenes = write(scenes, "\n".join(problems))
                problems = length_problems(sum(len(s.narration.split()) for s in scenes), len(scenes), goal, where)
            sections.append(scenes)
            warnings += problems

        script = Script.from_sections(outline.title, sections, [s["url"] for s in state.get("sources", [])])
        warnings += length_problems(script.word_count(), len(script.scenes), targets, "El vídeo")
        return {"script": script.model_dump(), "script_warnings": warnings}

    return write_script
