"""Your script and your images/videos (manual mode).

After write_script the run asks if you keep its script; [n] (or --manual, straight after the
outline) pauses (nodes/manual_script.py) with this folder ready in the run:

    <run>/manual/guion.md     your script (the outline's sections, with write_script's scenes if it ran)
    <run>/manual/visuales/    your images and videos

guion.md:
    # Title
    ## Any section heading            (a new section; the text after ## is only for you)
    One paragraph = one scene; scenes separated by a blank line.
    [file.jpg] before a word          the file (in visuales/) is seen from that word until the
                                      next mark of the scene (text before the first mark: the first file)
    // a line starting with // is ignored

A scene without any mark is searched automatically, like any other (manual and automatic
visuals can be mixed). Everything can still be changed at the review pause (<run>/montaje/).

Prompts per beat (written by beats on every run, manual or not): <run>/manual/prompts_imagenes.md
lists the video's brief and every beat (number, seconds, text, idea, what must and must not be
seen, searches, prompt); plan_visual.json has the same as JSON, to paste whole into your AI. An
image made with it goes into <run>/montaje/ with the beat's number (graph/montage.py).

Inbox: what you leave beforehand in data/manual/ (guion.md, visuales/) is copied into
<run>/manual/ when you write your script (never over a file already in the run).
"""
import json
import re
import shutil
from pathlib import Path

from graph.lengths import duration_text
from graph.nodes.brief import brief_lines
from graph.script_files import ROLE_LABELS
from graph.state import Scene, Script

FOLDER = "manual"
SCRIPT_FILE = "guion.md"
MEDIA_FOLDER = "visuales"
BEAT_PROMPTS = "prompts_imagenes.md"
PLAN_FILE = "plan_visual.json"   # the same, as JSON: the brief + every beat, to paste into your AI at once
INBOX = Path(__file__).resolve().parents[2] / "data" / "manual"   # files left before the run
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MARK = re.compile(r"\[([^\[\]]+\.[A-Za-z0-9]{2,4})\]")      # [file.ext]
ON_SCREEN = re.compile(r"^en pantalla\s*:", re.IGNORECASE)    # old on-screen text lines: skipped

INSTRUCTIONS = """\
// GUION MANUAL. Escribe la narración debajo de cada sección; lo que hay que contar está en outline.md.
//  - Una escena = un párrafo, de 1-2 frases. Separa las escenas con una línea en blanco.
//  - Tus imágenes y vídeos van en la carpeta visuales/ (junto a este archivo). Escribe
//    [nombre.jpg] justo antes de la palabra en la que debe aparecer: se ve hasta la siguiente
//    marca de la escena. Una escena SIN marcas: el pipeline busca sus imágenes solo, y
//    antes de montar puedes cambiar cualquiera (carpeta montaje/ del run).
//  - Formatos: vídeo (mp4, mov, webm, mkv) y foto (jpg, png, webp). Las fotos llevan zoom lento.
//  - Las líneas que empiezan por // se ignoran (puedes borrarlas).
//  - Duración objetivo: {duration} (unas {words} palabras: ~2,5 palabras por segundo).
//
// Ejemplo:
//   [cables.mp4] Los cables de cobre son hoy el gran freno de la inteligencia artificial.
//
//   Ocupan tanto que [servidores.jpg] los servidores apenas pueden respirar.
"""


def folder_of(run_folder):
    return Path(run_folder) / FOLDER


def kind_of(path):
    """video | photo | None (not a supported file)."""
    suffix = Path(path).suffix.lower()
    return "video" if suffix in VIDEO_EXTENSIONS else "photo" if suffix in PHOTO_EXTENSIONS else None


def prepare(run_folder, outline, targets, script=None):
    """Creates <run>/manual/ with what is in the inbox (data/manual/), the guion.md template and
    visuales/. With `script` (the one write_script made) the template already has its scenes: keep,
    edit or replace them. Never overwrites your files."""
    folder = folder_of(run_folder)
    (folder / MEDIA_FOLDER).mkdir(parents=True, exist_ok=True)
    copy_inbox(run_folder)
    path = folder / SCRIPT_FILE
    if not path.exists():
        lines = [INSTRUCTIONS.format(duration=duration_text(targets["seconds"]), words=targets["words"]),
                 f"# {script.title if script else outline.title}", ""]
        for n, s in enumerate(outline.sections, start=1):
            lines += [f"## {n}. [{ROLE_LABELS.get(s.role, s.role)}] {s.title}",
                      f"// {s.goal} · emoción: {s.emotion} · ~{s.words} palabras", ""]
            for scene in (c for c in script.scenes if c.section == n) if script else ():
                lines += [scene.narration, ""]
            lines += [""] if not script else []
        path.write_text("\n".join(lines), encoding="utf-8")
    return folder


def copy_inbox(run_folder):
    """data/manual/ -> <run>/manual/, without overwriting what is already in the run."""
    folder, copied = folder_of(run_folder), 0
    for source in INBOX.rglob("*") if INBOX.exists() else ():
        target = folder / source.relative_to(INBOX)
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    if copied:
        print(f"  · manual: {copied} archivos copiados de {INBOX}")


def write_beat_prompts(run_folder, beats, width, height, brief=None):
    """<run>/manual/prompts_imagenes.md (to read) and plan_visual.json (to paste whole into your AI):
    the brief and the beats searched automatically (not those from your marks)."""
    folder = folder_of(run_folder)
    folder.mkdir(parents=True, exist_ok=True)
    shape = "vertical" if height > width else "horizontal"
    wanted = [b for b in beats if b["kind"] != "manual"]                  # not your marks
    lines = [f"# Imágenes por fragmento ({width}x{height}, {shape})", "",
             "Para sustituir un plano por una imagen hecha con tu IA: en la pausa antes de montar, "
             "ponla en la carpeta montaje/ del run con el número del fragmento (05.jpg, 05.mp4...).",
             f"Todo junto, para pegarlo en tu IA: {PLAN_FILE}", ""]
    if brief:
        lines += ["## Brief visual (vale para todos los fragmentos)"] + brief_lines(brief) + [""]
    for b in wanted:
        lines += [f"## {b['id']:02d} · {b['seconds']:.1f} s · escena {b['scene']}", f"Texto: {b['text']}"]
        lines += [f"Idea: {b['idea']}"] if b.get("idea") else []
        lines += [f"Debe verse: {', '.join(b['must_show'])}"] if b.get("must_show") else []
        lines += [f"No debe verse: {', '.join(b['must_not_show'])}"] if b.get("must_not_show") else []
        lines += [f"Qué se ve: {b['subject']}", f"Búsquedas: {', '.join(b.get('queries', []))}",
                  f"Prompt: {b['prompt']}", ""]
    path = folder / BEAT_PROMPTS
    path.write_text("\n".join(lines), encoding="utf-8")
    plan = {"formato": {"width": width, "height": height, "orientation": shape},
            "brief": brief or {},
            "fragmentos": [{k: b[k] for k in ("id", "seconds", "scene", "text", "idea", "must_show", "must_not_show",
                                              "subject", "queries", "prompt") if k in b} for b in wanted]}
    (folder / PLAN_FILE).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse(run_folder, sources=()):
    """guion.md -> (Script, marks, problems).
    marks: {scene id: [{"word": index in the scene's narration, "path": absolute path}, ...]}
    problems: what must be fixed before going on (empty: all good)."""
    folder = folder_of(run_folder)
    path = folder / SCRIPT_FILE
    if not path.exists():
        return None, {}, [f"No existe {path}"]
    title, section, scenes, current = None, 0, [], None
    for raw in path.read_text(encoding="utf-8").splitlines() + [""]:
        line = raw.strip()
        if line.startswith("//") or ON_SCREEN.match(line):
            continue
        if not line:
            current = None                                          # a blank line closes the scene
        elif line.startswith("## "):
            section, current = section + 1, None
        elif line.startswith("# ") and title is None:
            title = line[2:].strip()
        else:
            if current is None:
                current = {"section": max(1, section), "lines": []}
                scenes.append(current)
            current["lines"].append(line)

    problems, result, marks = [], [], {}
    for n, s in enumerate((s for s in scenes if s["lines"]), start=1):
        words, scene_marks = [], []
        # split() with a group: text, mark, text, mark, ... -> odd pieces are marks
        for i, piece in enumerate(MARK.split(" ".join(s["lines"]))):
            if i % 2 == 0:
                words += piece.split()
            elif kind_of(piece):                                       # a mark: its word is the next one
                file = folder / MEDIA_FOLDER / piece.strip()
                if not file.exists():
                    problems.append(f"Escena {n}: no está el archivo {MEDIA_FOLDER}/{piece.strip()}")
                scene_marks.append({"word": len(words), "path": str(file)})
            else:
                problems.append(f"Escena {n}: formato no admitido [{piece}] "
                                "(vídeo: mp4, mov, webm, mkv; foto: jpg, png, webp)")
        if not words:
            problems.append(f"Escena {n}: tiene marcas pero no narración")
            continue
        for m in scene_marks:
            m["word"] = min(m["word"], len(words) - 1)                 # a mark at the very end: the last word
        scene_marks[:1] = [dict(scene_marks[0], word=0)] if scene_marks else []   # before the first: the first
        result.append(Scene(id=n, section=s["section"], narration=" ".join(words)))
        if scene_marks:
            marks[n] = scene_marks

    if not result:
        problems.append(f"El guion está vacío: escribe la narración en {path}")
    if problems:
        return None, {}, problems
    return Script(title=title or "Sin título", scenes=result, sources=list(dict.fromkeys(sources))), marks, []
