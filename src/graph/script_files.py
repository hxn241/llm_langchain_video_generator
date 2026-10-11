"""The angles, the outline and the script on disk, inside the run's folder:

    angles.json / outline.json / script.json   the source of truth (what you edit before approving)
    angles.md   / outline.md   / script.md     the same, easy to read (generated; edits are NOT read back)
"""
import json

from graph.state import AnglesDraft, Outline, Script

ROLE_LABELS = {"hook": "gancho", "context": "contexto", "tension": "tensión", "development": "desarrollo",
               "reveal": "revelación", "conclusion": "conclusión"}


def _write_json(path, model):
    path.write_text(json.dumps(model.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ angles
def save_angles(angles: AnglesDraft, folder):
    _write_json(folder / "angles.json", angles)
    (folder / "angles.md").write_text(angles_markdown(angles), encoding="utf-8")


def load_angles(folder):
    """angles.json -> AnglesDraft (validated: a hand edit that breaks the schema fails here, with the reason)."""
    return AnglesDraft.model_validate_json((folder / "angles.json").read_text(encoding="utf-8"))


def angles_markdown(angles, indent=""):
    lines = []
    for n, a in enumerate(angles.angles, start=1):
        lines += [f"{n}. {a.title}" + ("   <- recomendado" if n == angles.best else ""),
                  f"   Pregunta: {a.question}", f"   Gancho: \"{a.hook}\"", f"   Por qué engancha: {a.why}",
                  *([f"   Hechos: {', '.join(map(str, a.facts))}"] if a.facts else []), ""]
    lines.append(f"Recomendado: {angles.best}. {angles.reason}")
    return "\n".join(indent + line if line else line for line in lines) + "\n"


# ----------------------------------------------------------------- outline
def save_outline(outline: Outline, facts, folder):
    """Writes outline.json and outline.md (with the numbered facts, to see what each section uses)."""
    _write_json(folder / "outline.json", outline)
    text = outline_markdown(outline) + "\n## Hechos\n\n" + "\n".join(
        f"{n}. {f['fact']} ({f['source_url']})" for n, f in enumerate(facts, start=1)) + "\n"
    (folder / "outline.md").write_text(text, encoding="utf-8")


def load_outline(folder):
    """outline.json -> Outline (validated: a hand edit that breaks the schema fails here, with the reason)."""
    return Outline.model_validate_json((folder / "outline.json").read_text(encoding="utf-8"))


def outline_markdown(outline, indent=""):
    """Also accepts an OutlineDraft (no words yet)."""
    lines = [f"# {outline.title}", f"Hilo conductor: {outline.angle}", ""]
    for n, s in enumerate(outline.sections, start=1):
        words = getattr(s, "words", None)
        lines.append(f"{n}. [{ROLE_LABELS.get(s.role, s.role)}] {s.title}" + (f"  (~{words} palabras)" if words else ""))
        lines.append(f"   {s.goal}  · emoción: {s.emotion}"
                     + (f"  [hechos: {', '.join(map(str, s.facts))}]" if s.facts else ""))
        lines += [f"   ? {q}" for q in getattr(s, "questions", None) or []]
    return "\n".join(indent + line if line else line for line in lines) + "\n"


# ------------------------------------------------------------------ script
def save_script(script: Script, folder):
    """Writes script.json and script.md; returns their paths."""
    json_path, md_path = folder / "script.json", folder / "script.md"
    _write_json(json_path, script)
    md_path.write_text(script_markdown(script), encoding="utf-8")
    return json_path, md_path


def load_script(folder):
    """script.json -> Script (validated: a hand edit that breaks the schema fails here, with the reason)."""
    return Script.model_validate_json((folder / "script.json").read_text(encoding="utf-8"))


def script_markdown(script: Script):
    lines = [f"# {script.title}", "", f"{script.word_count()} palabras · {len(script.scenes)} escenas", ""]
    section = None
    for s in script.scenes:
        if s.section != section:
            section = s.section
            lines += [f"## Sección {section}", ""]
        lines += [f"### Escena {s.id}" + (" (gancho)" if s.id == 1 else ""), "", s.narration, ""]
    if script.sources:
        lines += ["## Fuentes", ""] + [f"- {url}" for url in script.sources]
    return "\n".join(lines) + "\n"
