"""Manual mode (run_pipeline.py --manual): YOU write the script and bring the images/videos.

Research, angle and outline are made as always; after the outline the graph pauses
(nodes/manual_script.py) with this folder ready in the run:

    <run>/manual/guion.md     your script (a template with the outline's sections)
    <run>/manual/visuales/    your images and videos

guion.md:
    # Title
    ## Any section heading            (a new section; the text after ## is only for you)
    One paragraph = one scene; scenes separated by a blank line.
    En pantalla: text                 (optional line in a scene: its on-screen text)
    [file.jpg] before a word          the file (in visuales/) is seen from that word until the
                                      next mark of the scene (text before the first mark: the first file)
    // a line starting with // is ignored

    <run>/manual/media_escenas.json   (optional) a Pexels page per scene, downloaded for you:
        [{"id": 1, "subject": "...", "media": {"page_url": "https://www.pexels.com/video/...",
                                               "alternative": "https://www.pexels.com/video/..."}}, ...]
        id = the scene's number (its paragraph in guion.md). fetch() downloads each page with
        modules/pexels_scraper.py (Playwright) to visuales/<id>_<file> (the alternative if the
        page fails; files already there are not downloaded again) and that file is seen during
        the whole scene, without writing marks. A scene with [file] marks uses its marks.

A scene without any mark (nor an entry in media_escenas.json) gets its visuals automatically
(beats + stock search), as in the automatic mode: manual and automatic visuals can be mixed.
"""
import json
import re
from pathlib import Path

from graph.lengths import duration_text
from graph.script_files import ROLE_LABELS
from graph.state import Scene, Script

FOLDER = "manual"
SCRIPT_FILE = "guion.md"
MEDIA_FOLDER = "visuales"
MEDIA_LIST = "media_escenas.json"
BROWSER = "msedge"            # the Edge of Windows: nothing to download (None = Playwright's own Chromium)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MARK = re.compile(r"\[([^\[\]]+\.[A-Za-z0-9]{2,4})\]")      # [file.ext]
ON_SCREEN = re.compile(r"^en pantalla\s*:\s*(.*)$", re.IGNORECASE)

INSTRUCTIONS = """\
// GUION MANUAL. Escribe la narración debajo de cada sección; lo que hay que contar está en outline.md.
//  - Una escena = un párrafo, de 1-2 frases. Separa las escenas con una línea en blanco.
//  - Texto en pantalla (opcional): una línea "En pantalla: tu texto" dentro de la escena.
//  - Tus imágenes y vídeos van en la carpeta visuales/ (junto a este archivo). Escribe
//    [nombre.jpg] justo antes de la palabra en la que debe aparecer: se ve hasta la siguiente
//    marca de la escena. Una escena SIN marcas: el pipeline busca sus imágenes solo.
//  - O deja media_escenas.json (junto a este archivo) con una página de Pexels por escena: se
//    descargan solas y cada una se ve durante su escena ("id" = número de escena, por orden).
//  - Formatos: vídeo (mp4, mov, webm, mkv) y foto (jpg, png, webp). Las fotos llevan zoom lento.
//  - Las líneas que empiezan por // se ignoran (puedes borrarlas).
//  - Duración objetivo: {duration} (unas {words} palabras: ~2,5 palabras por segundo).
//
// Ejemplo:
//   [cables.mp4] Los cables de cobre son hoy el gran freno de la inteligencia artificial.
//   En pantalla: El cobre, al límite
//
//   Ocupan tanto que [servidores.jpg] los servidores apenas pueden respirar.
"""


def folder_of(run_folder):
    return Path(run_folder) / FOLDER


def kind_of(path):
    """video | photo | None (not a supported file)."""
    suffix = Path(path).suffix.lower()
    return "video" if suffix in VIDEO_EXTENSIONS else "photo" if suffix in PHOTO_EXTENSIONS else None


def prepare(run_folder, outline, targets):
    """Creates <run>/manual/ with the guion.md template and visuales/. Never overwrites your files."""
    folder = folder_of(run_folder)
    (folder / MEDIA_FOLDER).mkdir(parents=True, exist_ok=True)
    path = folder / SCRIPT_FILE
    if not path.exists():
        lines = [INSTRUCTIONS.format(duration=duration_text(targets["seconds"]), words=targets["words"]),
                 f"# {outline.title}", ""]
        for n, s in enumerate(outline.sections, start=1):
            lines += [f"## {n}. [{ROLE_LABELS.get(s.role, s.role)}] {s.title}",
                      f"// {s.goal} · emoción: {s.emotion} · ~{s.words} palabras", "", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
    return folder


def media_entries(run_folder):
    """media_escenas.json -> ({scene id: {"page_url", "alternative"}}, problems). Empty if there is none."""
    path = folder_of(run_folder) / MEDIA_LIST
    if not path.exists():
        return {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        return {}, [f"{MEDIA_LIST} no es un JSON válido: {e}"]
    entries, problems = {}, []
    for item in data if isinstance(data, list) else []:
        media = item.get("media") or {}
        if not str(item.get("id", "")).isdigit() or not (media.get("page_url") or item.get("page_url")):
            problems.append(f"{MEDIA_LIST}: una entrada sin \"id\" numérico o sin \"page_url\": {item}")
            continue
        entries[int(item["id"])] = {"page_url": media.get("page_url") or item["page_url"],
                                    "alternative": media.get("alternative")}
    return entries, problems


def downloaded(run_folder, scene_id):
    """The file of media_escenas.json downloaded for this scene (visuales/<id>_...), or None."""
    hits = [p for p in (folder_of(run_folder) / MEDIA_FOLDER).glob(f"{scene_id:02d}_*")
            if kind_of(p) and p.stat().st_size > 0]
    return hits[0] if hits else None


def fetch(run_folder):
    """Downloads what media_escenas.json lists and is not in visuales/ yet (the alternative if the
    page fails). Returns the problems (pages that could not be downloaded). Needs Playwright + Chromium."""
    entries, problems = media_entries(run_folder)
    missing = {n: e for n, e in entries.items() if not downloaded(run_folder, n)}
    if not missing:
        return problems
    try:
        from modules import pexels_scraper as scraper            # Playwright: only needed here
    except ImportError as e:
        return problems + [f"Para descargar {MEDIA_LIST} hace falta Playwright ({e})"]
    media_dir = folder_of(run_folder) / MEDIA_FOLDER
    print(f"  · manual: descargando {len(missing)} archivos de {MEDIA_LIST}...")
    for attempt in ("page_url", "alternative"):
        jobs = [scraper.Job(e[attempt], f"{n:02d}") for n, e in missing.items()
                if e.get(attempt) and not downloaded(run_folder, n)]
        if not jobs:
            continue
        try:
            scraper.download_all(jobs, media_dir, channel=BROWSER)
        except Exception as e:                                   # e.g. the browser could not start
            return problems + [f"No se pudo descargar {MEDIA_LIST}: {str(e)[:300]}"]
    for n, e in missing.items():
        if downloaded(run_folder, n):
            print(f"  · manual: escena {n} <- {downloaded(run_folder, n).name}")
        else:
            problems.append(f"Escena {n}: no se pudo descargar {e['page_url']}"
                            + (f" ni {e['alternative']}" if e.get("alternative") else "")
                            + f" (descárgalo tú a {MEDIA_FOLDER}/{n:02d}_nombre.mp4)")
    return problems


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
        if line.startswith("//"):
            continue
        if not line:
            current = None                                          # a blank line closes the scene
        elif line.startswith("## "):
            section, current = section + 1, None
        elif line.startswith("# ") and title is None:
            title = line[2:].strip()
        elif ON_SCREEN.match(line):
            if current is None:
                current = {"section": max(1, section), "lines": [], "on_screen": None}
                scenes.append(current)
            current["on_screen"] = ON_SCREEN.match(line).group(1).strip() or None
        else:
            if current is None:
                current = {"section": max(1, section), "lines": [], "on_screen": None}
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
        result.append(Scene(id=n, section=s["section"], narration=" ".join(words), on_screen_text=s["on_screen"]))
        if scene_marks:
            marks[n] = scene_marks

    # media_escenas.json: its file for each scene without marks of its own, the whole scene
    entries, entry_problems = media_entries(run_folder)
    problems += entry_problems
    for n in sorted(entries):
        if n > len(result):
            problems.append(f"{MEDIA_LIST} tiene la escena {n}, pero el guion tiene {len(result)} escenas")
        elif n not in marks:
            file = downloaded(run_folder, n)
            if file:
                marks[n] = [{"word": 0, "path": str(file)}]
            else:
                problems.append(f"Escena {n}: falta su archivo de {MEDIA_LIST} en {MEDIA_FOLDER}/ "
                                f"(se descarga al responder [y]; o déjalo tú como {n:02d}_nombre.mp4)")
    if not result:
        problems.append(f"El guion está vacío: escribe la narración en {path}")
    if problems:
        return None, {}, problems
    return Script(title=title or "Sin título", scenes=result, sources=list(dict.fromkeys(sources))), marks, []
