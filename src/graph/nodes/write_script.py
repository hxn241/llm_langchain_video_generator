"""write_script 🤖: approved outline (+ chosen angle) + facts -> Script, ONE SECTION PER CALL.

A 9B model cannot write 10 minutes (~1,600 words) well in one go: it shortens,
repeats itself or loses the thread. Each section is written on its own, seeing
the whole outline, its own facts and how the previous section ended.

- Each section's words come from the outline (lengths.assign_words) and its scenes
  from the format. If a section misses them, or a scene repeats an expression (or the
  on-screen text) of the scene before it, it is asked ONCE more saying what to fix.
- With review_feedback (from the review node) each section is rewritten from its
  previous version applying it.
- Each section sees the facts the sections before it already told (not to tell them again), and
  in long videos the last one sees how the video started (to close coming back to it).
- The video's total length is the outline's: after deep_research a long video with less
  material plans fewer words (lengths.planned_targets), and it is checked against that.
- The sources are the articles of the research: the LLM never writes them.
"""
import json

from graph.lengths import length_problems, length_targets, planned_targets, section_targets
from graph.nodes.align import normalize
from graph.script_files import ROLE_LABELS
from graph.state import Outline, Script, SectionDraft
from prompts import load_prompt
from services.settings import load_settings

NODE = "write_script"
FORMAT_LABELS = {"short": "Short vertical (9:16)", "long": "vídeo horizontal (16:9)"}
CONTEXT_SCENES = 2            # scenes of the previous section shown for continuity


def section_request(state, outline, number, targets, facts, video_format, before=None,
                    previous=None, feedback=None, opening=None):
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
    if video_format == "long":
        parts.append("Es un VÍDEO LARGO: aplica también las técnicas de \"Vídeos largos\".")
    parts += [f"Longitud: entre {targets['min_scenes']} y {targets['max_scenes']} escenas, y entre "
              f"{targets['min_words']} y {targets['max_words']} palabras de narración (objetivo: {targets['words']}).",
              "", "# Hechos para esta sección",
              "\n".join(f"{n}. {facts[n - 1]['fact']}" for n in used)]
    told = sorted({f for s in outline.sections[:number - 1] for f in s.facts} - set(section.facts))
    if told:
        parts += ["", "# Ya contados en secciones anteriores (no los vuelvas a contar; como mucho, alúdelos)",
                  "\n".join(f"{n}. {facts[n - 1]['fact']}" for n in told if 1 <= n <= len(facts))]
    if opening and number == total and video_format == "long":
        parts += ["", "# Así empezó el vídeo (el cierre puede volver a esta imagen o pregunta)", opening]
    if before:
        parts += ["", "# Así termina la sección anterior (enlaza con ella, no la repitas)",
                  "\n".join(s.narration for s in before)]
    if previous:
        parts += ["", "# Versión anterior de esta sección",
                  json.dumps([s.model_dump(exclude={"id", "section"}) for s in previous], ensure_ascii=False, indent=1)]
    if feedback:
        parts += ["", "# Corrige esto (reescribe la sección aplicándolo)", feedback]
    return "\n".join(parts)


def trigrams(text):
    """Groups of 3 consecutive words with at least 2 content words (4+ letters): 'fotónica de
    silicio' counts, 'de la vivienda' does not (too common to be a repetition worth fixing)."""
    words = [w for w in (normalize(w) for w in text.split()) if w]
    return {" ".join(g) for g in zip(words, words[1:], words[2:]) if sum(len(w) >= 4 for w in g) >= 2}


def repetition_problems(scenes, before=None):
    """What a scene repeats from the one just before it (the first one, from the end of the previous
    section): the same expression."""
    problems = []
    chain = ([before[-1]] if before else []) + list(scenes)
    for previous, scene in zip(chain, chain[1:]):
        repeated = trigrams(previous.narration) & trigrams(scene.narration)
        if repeated:
            problems.append(f"\"{scene.narration[:60]}...\" repite \"{sorted(repeated)[0]}\" de la escena "
                            "anterior: refiérete a ello de otra forma (sinónimo, pronombre, descripción o "
                            "metáfora) o cuenta algo nuevo")
    return problems


def make_write_script(hub, settings=None):
    """The node, bound to an LLMHub (its model for this node comes from config.yaml: llm.write_script)."""
    settings = settings or load_settings()
    generate = hub.chain(load_prompt(NODE), schema=SectionDraft, node=NODE)

    def write_script(state):
        outline, facts = Outline.model_validate(state["outline"]), state["facts"]
        video_format = state.get("format") or settings.get("format", "long")
        targets = planned_targets(length_targets(video_format, settings), sum(s.words for s in outline.sections))
        # Rewriting after a review: each section starts from its previous version and the feedback
        old = Script.model_validate(state["script"]) if state.get("script") else None
        feedback = state.get("review_feedback") if old else None

        sections, warnings = [], []
        for number, section in enumerate(outline.sections, start=1):
            goal = section_targets(section.words, targets["words_per_scene"])
            before = sections[-1][-CONTEXT_SCENES:] if sections else None
            previous = old.section_scenes(number) if old else None

            opening = " ".join(s.narration for s in sections[0][:2]) if sections else None

            def write(previous, feedback):
                return generate.invoke({"text": section_request(state, outline, number, goal, facts, video_format,
                                                                before, previous, feedback, opening)}).scenes

            def check(scenes):
                return (length_problems(sum(len(s.narration.split()) for s in scenes), len(scenes), goal,
                                        f"La sección {number}") + repetition_problems(scenes, before))

            scenes = write(previous, feedback)
            problems = check(scenes)
            if problems:                                    # one retry, saying exactly what to fix
                scenes = write(scenes, "\n".join(problems))
                problems = check(scenes)
            sections.append(scenes)
            warnings += problems

        script = Script.from_sections(outline.title, sections, [s["url"] for s in state.get("sources", [])])
        warnings += length_problems(script.word_count(), len(script.scenes), targets, "El vídeo")
        return {"script": script.model_dump(), "script_warnings": warnings}

    return write_script
