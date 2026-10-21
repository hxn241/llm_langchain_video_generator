"""write_script 🤖: approved outline (+ chosen angle) + facts -> Script.

- Short, and long up to script.whole_max_words (5-6 min, ~900 words): the whole script in ONE call.
  A video is one arc (hook, the "ah, so that's why", a close that comes back to the hook): written
  in pieces, each section repeated the others' ideas (2026-10-15, butterflies: "the bird sees it
  perfectly" in 4 sections) and a short ended with a generic question. Its rewrites are whole too.
- Longer: ONE SECTION PER CALL. Ten minutes (~1,600 words) in one go came out shortened, repeated
  or lost: each section is written on its own, seeing the whole outline, its own facts and how
  the previous section ended. It also sees the facts the sections before it already told (not
  to tell them again), and the last one sees how the video started (to close coming back to it).

- The words come from the outline (lengths.assign_words) and the scenes from the format. If the
  script (or a section) misses them, or a scene repeats an expression of the scene before it,
  it is asked ONCE more saying what to fix.
- Rewriting: from the previous version. Your reason at the choose_script pause (review_feedback)
  applies to everything; review_script's notes (script_notes) say their scene: in a long video
  only the sections with a note are written again (a note about the whole video, scene 0, rewrites
  them all).
- The video's total length is the outline's: after deep_research a video with less material
  plans fewer words (lengths.planned_targets), and it is checked against that.
- The sources are the articles of the research: the LLM never writes them.
- The model writes a PLAN before the scenes (the question it opens, where it gets complicated and
  solved, the new hooks, where the strong fact goes, the close): prompts/write_script.md, "Que no se
  vaya nadie". It is not narrated: <run>/plan_guion.md, to read next to script.md.
- What a rewrite was asked (your reason, the reviewer's notes) goes to the state (script_requests):
  review_script checks it was done.
- Long videos present themselves right after the hook, in one of five forms that rotate from
  video to video (graph/intros.py): the request says which.
"""
import json

from graph import intros
from graph.lengths import length_problems, length_targets, planned_targets, section_targets
from graph.nodes.align import normalize
from graph.script_files import ROLE_LABELS, TYPE_LABELS
from graph.state import Outline, Scene, SceneDraft, Script, ScriptDraft, SectionPlanDraft
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "write_script"
FORMAT_LABELS = {"short": "Short vertical (9:16)", "long": "vídeo horizontal (16:9)"}
CONTEXT_SCENES = 2            # scenes of the previous section shown for continuity


def video_lines(state, outline, video_format):
    """What every request starts with: format, kind of video, title, guiding question, angle, notes."""
    angle = state.get("angle") or {}
    parts = [f"Formato: {FORMAT_LABELS[video_format]}."]
    if state.get("video_type"):
        parts.append(f"Tipo de vídeo: {TYPE_LABELS[state['video_type']]}.")
    parts += [f"Título del vídeo: {outline.title}", f"Hilo conductor: {outline.angle}"]
    if angle.get("title"):
        parts.append(f"Enfoque: {angle['title']}")
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    return parts


def fact_lines(facts, numbers):
    return "\n".join(f"{n}. {facts[n - 1]['fact']}" for n in numbers if 1 <= n <= len(facts))


def scene_json(scenes):
    """A previous version for the request, with each scene's number in the video (the notes use it)."""
    return [({"escena": s.id} if isinstance(s, Scene) else {}) | {"narración": s.narration} for s in scenes]


def length_line(targets):
    return (f"Longitud: entre {targets['min_scenes']} y {targets['max_scenes']} escenas, y entre "
            f"{targets['min_words']} y {targets['max_words']} palabras de narración (objetivo: {targets['words']}).")


def script_request(state, outline, targets, facts, video_format, previous=None, feedback=None, intro=None):
    """The whole script in one call (shorts, and long videos up to script.whole_max_words).
    intro: a long video's presentation form (graph/intros.py)."""
    angle = state.get("angle") or {}
    plan = "\n".join(f"{n}. [{ROLE_LABELS[s.role]}] {s.title} (~{s.words} palabras): {s.goal} Emoción: {s.emotion}."
                     + (f" Hechos: {', '.join(map(str, s.facts))}." if s.facts else "")
                     for n, s in enumerate(outline.sections, start=1))
    used = sorted({f for s in outline.sections for f in s.facts}) or range(1, len(facts) + 1)
    parts = video_lines(state, outline, video_format) + [
        "", "# Esquema", plan, "",
        f"# Escribe el guion completo: las {len(outline.sections)} secciones del esquema, en su orden"]
    if angle.get("hook"):
        parts.append(f"Gancho propuesto (úsalo o mejóralo, sin salirte de los hechos): \"{angle['hook']}\"")
    if video_format == "long":
        parts.append("Es un VÍDEO LARGO: aplica también las técnicas de \"Vídeos largos\". Cada sección, con las "
                     "palabras que le da el esquema. Lo que ya ha dicho una sección no lo repite otra: ni los "
                     "datos ni las ideas.")
    if intro:
        parts += intros.request_lines(intro)
    parts += [length_line(targets), "", "# Hechos", fact_lines(facts, used)]
    if previous:
        sections = sorted({s.section for s in previous})
        parts += ["", "# Versión anterior del guion", json.dumps(
            [{"sección": n, "escenas": scene_json([s for s in previous if s.section == n])} for n in sections],
            ensure_ascii=False, indent=1)]
    if feedback:
        parts += ["", "# Corrige esto (reescribe el guion aplicándolo)",
                  "Cambia SOLO lo que piden estas correcciones (y lo justo para que las escenas de alrededor "
                  "sigan encajando); el resto de escenas, copiadas tal cual de la versión anterior. Cada "
                  "cambio que no se pide puede romper algo que estaba bien.", feedback]
    return "\n".join(parts)


def section_request(state, outline, number, targets, facts, video_format, before=None,
                    previous=None, feedback=None, opening=None, intro=None):
    """One section (long videos). intro: the presentation form, for the first section."""
    section = outline.sections[number - 1]
    total = len(outline.sections)
    plan = "\n".join(f"{n}. [{ROLE_LABELS[s.role]}] {s.title}: {s.goal}"
                     + ("   <- ESTA SECCIÓN" if n == number else "")
                     for n, s in enumerate(outline.sections, start=1))
    position = ("Es la PRIMERA sección: la primera escena es el gancho." if number == 1 else
                "Es la ÚLTIMA sección: cierra respondiendo la pregunta del gancho, sin resumir." if number == total else
                "Es una sección intermedia.")
    used = section.facts or range(1, len(facts) + 1)                 # no facts listed: all of them
    angle = state.get("angle") or {}
    parts = video_lines(state, outline, video_format)
    parts += ["", "# Esquema", plan, "",
              f"# Escribe la sección {number} de {total}: {section.title}", section.goal,
              f"Función en la historia: {ROLE_LABELS[section.role]}. Emoción que busca: {section.emotion}.",
              position]
    if number == 1 and angle.get("hook"):
        parts.append(f"Gancho propuesto (úsalo o mejóralo, sin salirte de los hechos): \"{angle['hook']}\"")
    if video_format == "long":
        parts.append("Es un VÍDEO LARGO: aplica también las técnicas de \"Vídeos largos\".")
    if intro and number == 1:
        parts += intros.request_lines(intro)
    parts += [length_line(targets), "", "# Hechos para esta sección", fact_lines(facts, used)]
    told = sorted({f for s in outline.sections[:number - 1] for f in s.facts} - set(section.facts))
    if told:
        parts += ["", "# Ya contados en secciones anteriores (no los vuelvas a contar; como mucho, alúdelos)",
                  fact_lines(facts, told)]
    if opening and number == total and video_format == "long":
        parts += ["", "# Así empezó el vídeo (el cierre puede volver a esta imagen o pregunta)", opening]
    if before:
        parts += ["", "# Así termina la sección anterior (enlaza con ella, no la repitas)",
                  "\n".join(s.narration for s in before)]
    if previous:
        parts += ["", "# Versión anterior de esta sección",
                  json.dumps(scene_json(previous), ensure_ascii=False, indent=1)]
    if feedback:
        parts += ["", "# Corrige esto (reescribe la sección aplicándolo)", feedback]
    return "\n".join(parts)


def note_line(note):
    return (f"- {'Todo el vídeo' if note['scene'] == 0 else 'Escena ' + str(note['scene'])}: {note['problem']} "
            f"Arreglo: {note['fix']}")


def feedback_text(reason, notes, scene_ids=None, review=None):
    """Your reason (for everything, with the last review next to it: "aplica las notas" or "no uses
    el hecho 5" refer to it) + the reviewer's notes about these scenes (all if None) and about the
    whole video (scene 0)."""
    parts = []
    if reason and reason.strip():
        parts.append(reason.strip())
        if review and (review.get("issues") or review.get("dubious")):
            parts.append("Para entender lo que pide el autor, la última revisión del guion decía:\n" + "\n".join(
                [note_line(n) for n in review.get("issues", [])]
                + [f"- Hecho {d['fact']} dudoso: {d['why']}" for d in review.get("dubious", [])]))
    picked = [n for n in notes if scene_ids is None or n["scene"] == 0 or n["scene"] in scene_ids]
    if picked:
        parts.append("Notas del editor:\n" + "\n".join(note_line(n) for n in picked))
    return "\n\n".join(parts)


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
    prompt = load_prompt(NODE)
    write_whole = hub.chain(prompt, schema=ScriptDraft, node=NODE)
    write_section = hub.chain(prompt, schema=SectionPlanDraft, node=NODE)

    def whole_script(state, outline, targets, facts, video_format, old, feedback, plans, intro):
        """The whole script in one call -> ([scenes per section], warnings). plans[0]: its plan."""
        def write(previous, feedback):
            draft = write_whole.invoke({"text": script_request(state, outline, targets, facts, video_format,
                                                               previous, feedback, intro)})
            plans[0] = draft.plan.strip()
            return draft.sections

        def check(drafts):
            scenes = [s for d in drafts for s in d.scenes]
            problems = length_problems(sum(len(s.narration.split()) for s in scenes), len(scenes), targets,
                                       "El guion") + repetition_problems(scenes)
            if len(drafts) != len(outline.sections):
                problems.append(f"El guion tiene {len(drafts)} secciones y deben ser {len(outline.sections)}, "
                                "las del esquema y en su orden.")
            return problems

        drafts = write(old.scenes if old else None, feedback)
        problems = check(drafts)
        if problems:                                    # one retry, saying exactly what to fix
            made = Script.from_sections(outline.title, [d.scenes for d in drafts], [])
            drafts = write(made.scenes, "\n".join(problems))
            problems = check(drafts)
        return [d.scenes for d in drafts], problems

    def by_sections(state, outline, targets, facts, video_format, old, reason, notes, plans, intro):
        """Long videos: one call per section -> ([scenes per section], warnings). plans[n]: section n's plan."""
        sections, warnings = [], []
        for number, section in enumerate(outline.sections, start=1):
            goal = section_targets(section.words, targets["words_per_scene"])
            before = sections[-1][-CONTEXT_SCENES:] if sections else None
            opening = " ".join(s.narration for s in sections[0][:2]) if sections else None
            previous = old.section_scenes(number) if old else None
            feedback = (feedback_text(reason, notes, {s.id for s in previous}, state.get("script_review"))
                        if previous else None)
            if previous and not feedback:               # rewriting, and nothing to fix here: it stays
                sections.append([SceneDraft(narration=s.narration) for s in previous])
                continue

            def write(previous, feedback):
                draft = write_section.invoke({"text": section_request(state, outline, number, goal, facts,
                                                                      video_format, before, previous, feedback,
                                                                      opening, intro)})
                plans[number] = draft.plan.strip()
                return draft.scenes

            def check(scenes):
                return (length_problems(sum(len(s.narration.split()) for s in scenes), len(scenes), goal,
                                        f"La sección {number}") + repetition_problems(scenes, before))

            scenes = write(previous, feedback)
            problems = check(scenes)
            if problems:                                # one retry, saying exactly what to fix
                scenes = write(scenes, "\n".join(problems))
                problems = check(scenes)
            sections.append(scenes)
            warnings += problems
        return sections, warnings

    def write_script(state):
        outline, facts = Outline.model_validate(state["outline"]), state["facts"]
        video_format = state.get("format") or settings.get("format", "long")
        targets = planned_targets(length_targets(video_format, settings), sum(s.words for s in outline.sections))
        # Rewriting (your reason at the pause, or the reviewer's notes): from the previous version
        reason, notes = state.get("review_feedback") or "", state.get("script_notes") or []
        old = Script.model_validate(state["script"]) if state.get("script") and (reason or notes) else None
        requests = feedback_text(reason, notes, review=state.get("script_review")) if old else ""
        plans = {}
        # in one call: a short, or a long video short enough (config.yaml: script.whole_max_words)
        whole = (video_format == "short"
                 or targets["words"] <= (settings.get("script") or {}).get("whole_max_words", 1200))

        # a long video presents itself after the hook, each time in another form (graph/intros.py)
        intro = intros.form_for(state["run_id"]) if video_format == "long" and state.get("run_id") else None
        if intro and not old:
            print(f"  · write_script: presentación \"{intro['name']}\" tras el gancho")

        if whole:
            sections, warnings = whole_script(state, outline, targets, facts, video_format, old,
                                              requests or None, plans, intro)
        else:
            sections, warnings = by_sections(state, outline, targets, facts, video_format, old, reason, notes,
                                             plans, intro)
        script = Script.from_sections(outline.title, sections, [s["url"] for s in state.get("sources", [])])
        if not whole:                                   # a whole script was already checked as a whole
            warnings += length_problems(script.word_count(), len(script.scenes), targets, "El vídeo")
        # A long video's sections not written again keep their plan from before
        old_plans = state.get("script_plan") or ""
        plan = (plans.get(0, "") if whole else
                "\n\n".join(f"Sección {n}: {plans[n]}" if n in plans else
                            next((p for p in old_plans.split("\n\n") if p.startswith(f"Sección {n}:")), "")
                            for n in range(1, len(outline.sections) + 1)).strip())
        if state.get("run_id"):
            (runs_dir(state["run_id"]) / "plan_guion.md").write_text(
                f"# Plan del guionista\n\n{plan}\n", encoding="utf-8")
        return {"script": script.model_dump(), "script_warnings": warnings, "review_feedback": "", "script_notes": [],
                "script_plan": plan, "script_requests": requests}

    return write_script
