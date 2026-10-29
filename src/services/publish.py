"""A finished run -> a folder ready to upload (scripts/prepare_upload.py), OUTSIDE the pipeline:
it only reads what the run left (its files and its saved state) and never runs the graph.

    data/listo_para_subir/<20261011_tema>/
        video.mp4         the newest assembled video of the run (a copy)
        metadata.json     title, tags, category, privacy, publish_at, AI content... (edit it)
        descripcion.md    the description (edit it): summary, chapters (long videos), sources,
                          the music's credits (music_credits: from data/musica/), credits of the
                          stock images (Wikimedia's CC BY requires them), hashtags
        miniatura.jpg     1280x720: an image of the video (montaje/01 by default) + 2-4 big words
        subtitulos.srt    from the align node's word times: the exact narration, already timed

Only the title, summary, tags, hashtags and thumbnail words come from a model (llm.publish in
config.yaml, one call: prompts/publish.md); the rest is put together here from the run.
AI content ("synthetic_media") is true when montaje/ has a generated image or a file that does
not come from stock (yours: unknown, so it is marked; set it to false if it is not AI).
"""
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from pydantic import Field

from graph import manual, montage
from graph.nodes.assemble import scene_frames
from graph.state import Draft, Script
from graph.subtitles import lines as subtitle_lines
from llmkit import config
from prompts import load_prompt
from services import media
from services.settings import load_settings, runs_dir

NODE = "publish"
THUMB_SIZE = (1280, 720)
# long videos' thumbnail (side_thumbnail): text on the quieter half over a dark gradient
THUMB_TEXT_WIDTH = 0.5            # the text takes at most this share of the width
THUMB_SHADE = 0.65                # the gradient covers this share of the width...
THUMB_SHADE_ALPHA = 190           # ... and is this dark (0-255) at the edge
THUMB_HIGHLIGHT = "#FFD400"       # the key word
THUMB_MAX_FONT = 200              # px: one short word must not fill the whole image
THUMB_CONTRAST, THUMB_COLOR = 1.08, 1.12
DESCRIPTION_MAX = 5000            # YouTube's limit
TITLE_MAX = 100
SUBTITLE_WORDS = 7                # at most this many words per subtitle line
SUBTITLE_SECONDS = 3.0            # ... and this long
MIN_CHAPTER_SECONDS = 10          # YouTube ignores chapters shorter than this (and needs 3+)
NOT_VIDEOS = {"video_only.mp4"}   # intermediate files of assemble


class PublishDraft(Draft):
    title: str = Field(description="Título del vídeo, máximo 70 caracteres, sin clickbait engañoso")
    summary: str = Field(description="2-3 frases para la descripción: de qué va y qué se aprende")
    tags: list[str] = Field(min_length=3, max_length=20, description="8-15 etiquetas cortas, sin #")
    hashtags: list[str] = Field(min_length=1, max_length=3, description="2-3 hashtags que empiezan por #")
    thumbnail_text: str = Field(description="2-3 palabras en MAYÚSCULAS para la miniatura, que no repiten el título")
    thumbnail_highlight: str = Field(default="", description="La palabra de thumbnail_text que va en amarillo")


# ------------------------------------------------------------------ the run
def run_state(run_id):
    """The run's last saved state (read only: the checkpoint is not changed)."""
    path = config.STATE_DIR / "checkpoints.sqlite"
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    try:
        saved = SqliteSaver(conn).get_tuple({"configurable": {"thread_id": run_id}})
    finally:
        conn.close()
    if not saved:
        raise SystemExit(f"Run {run_id!r} not found in {path}")
    return saved.checkpoint["channel_values"]


def newest_video(run_folder):
    videos = [p for p in Path(run_folder).glob("*.mp4") if p.name not in NOT_VIDEOS]
    if not videos:
        raise SystemExit(f"{run_folder} has no assembled video yet (run the pipeline to the end)")
    return max(videos, key=lambda p: p.stat().st_mtime)


def scene_starts(state, settings):
    """Seconds where each scene starts in the video, exactly as assemble cut them."""
    script = Script.model_validate(state["script"])
    ids = [s.id for s in script.scenes]
    fps = settings["video"]["fps"]
    frames = scene_frames(state["audio"], ids, fps, settings["video"].get("scene_gap", 0.15))
    starts, t = {}, 0.0
    for i in ids:
        starts[i], t = t, t + frames[i] / fps
    return starts


def has_ai_content(run_folder, state):
    """True if montaje/ has a generated image or a file that is not one visuals took from stock."""
    folder = montage.folder_of(run_folder)
    manifest_path = folder / montage.MANIFEST
    ours = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for p in folder.iterdir() if folder.is_dir() else []:
        if not p.is_file() or not manual.kind_of(p) or not montage.NUMBER.match(p.stem):
            continue
        if p.name not in ours:
            return True                                    # yours: maybe AI, so it is marked
        visual = montage.visual_of(p.name, state.get("visuals"), state.get("visual_extras")) or {}
        if visual.get("ai_generated") or visual.get("source") == "generate":
            return True
    return False


# ------------------------------------------------------------------ pieces
def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def chapters(state, starts):
    """'0:00 Title' per outline section, for long videos; [] if YouTube would not take them."""
    script = Script.model_validate(state["script"])
    titles = [s["title"] for s in (state.get("outline") or {}).get("sections", [])]
    first = {}
    for s in script.scenes:
        first.setdefault(s.section, starts[s.id])
    marks = [(first[n], titles[n - 1]) for n in sorted(first) if 0 < n <= len(titles)]
    if len(marks) < 3 or marks[0][0] > 0:
        return []
    ends = [t for t, _ in marks[1:]] + [float("inf")]
    if any(end - t < MIN_CHAPTER_SECONDS for (t, _), end in zip(marks, ends)):
        return []
    return [f"{clock(t)} {title}" for t, title in marks]


def srt_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def subtitles(state, starts):
    """SRT from the narration's word times: short lines, cut at punctuation, never across scenes."""
    cues = [(start + line[0]["start"], start + line[-1]["end"], " ".join(x["word"] for x in line))
            for sid, start in starts.items()
            for line in subtitle_lines(state["audio"][sid].get("words") or [], SUBTITLE_WORDS, SUBTITLE_SECONDS)]
    out = []
    for n, (a, b, text) in enumerate(cues, start=1):
        b = min(b, cues[n][0]) if n < len(cues) else b              # never over the next line
        out += [str(n), f"{srt_time(a)} --> {srt_time(max(b, a + 0.3))}", text, ""]
    return "\n".join(out)


def source_line(s):
    """'- elpais.com (2026-10-01): Title https://...'"""
    head = " ".join(x for x in (s.get("outlet"), f"({s['date']})" if s.get("date") else None) if x)
    return f"- {head + ': ' if head else ''}{s.get('title') or ''} {s['url']}".replace("  ", " ")


def description(draft, chapter_lines, sources, credit_list, synthetic, music_lines=None):
    """Summary, chapters, sources, music credits (as they are: the licence forbids changing them),
    image credits, AI notice and hashtags. Over YouTube's 5,000 characters, the last image credits
    are left out (they also stay in the run's creditos.txt); the music's never."""
    credits = credit_list
    def build(credit_lines):
        parts = [draft.summary.strip()]
        if chapter_lines:
            parts += ["", *chapter_lines]
        if sources:
            parts += ["", "Fuentes:"] + [source_line(s) for s in sources]
        if music_lines:
            parts += ["", *music_lines]
        if credit_lines:
            parts += ["", "Imágenes y vídeos:"] + [f"- {c}" for c in credit_lines]
        if synthetic:
            parts += ["", "Algunas imágenes de este vídeo están generadas con inteligencia artificial."]
        parts += ["", " ".join(h if h.startswith("#") else f"#{h}" for h in draft.hashtags)]
        return "\n".join(parts)
    while len(build(credits)) > DESCRIPTION_MAX and credits:
        credits = credits[:-1]
    return build(credits)[:DESCRIPTION_MAX]


def credits(run_folder, state):
    """'author · licence · link' of each stock file that is IN montaje/ now (not of those you
    replaced): Wikimedia's CC BY / CC BY-SA require it; Pixabay's is a courtesy."""
    folder = montage.folder_of(run_folder)
    manifest_path = folder / montage.MANIFEST
    ours = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    lines = []
    for name in sorted(ours):
        if not (folder / name).exists():
            continue
        v = montage.visual_of(name, state.get("visuals"), state.get("visual_extras")) or {}
        if v.get("source_url"):
            lines.append(" · ".join(x for x in (v.get("author") or v.get("provider"), v.get("license"),
                                                v["source_url"]) if x))
    return list(dict.fromkeys(lines))


def music_credits(run_folder, library):
    """The credits of the run's music (montaje/musica.mp3), line by line; [] if it has none.
    From montaje/musica.txt if you left one; otherwise from the music library (`library`,
    youtube.music_dir): the track there with the same bytes, and its .txt next to it
    (data/musica/timeless.mp3 + timeless.txt). A video with music and no credits is not prepared:
    the free licences (CC BY...) require them, and without them the video can be claimed."""
    music = montage.music_of(run_folder)
    if not music:
        return []
    own = music.with_suffix(".txt")
    if not own.exists():
        digest = hashlib.md5(music.read_bytes()).hexdigest()
        own = next((p.with_suffix(".txt") for p in sorted(Path(library).glob("*"))
                    if p.suffix.lower() in montage.AUDIO_EXTENSIONS and hashlib.md5(p.read_bytes()).hexdigest() == digest), None)
    if not own or not own.exists():
        raise SystemExit(f"{music.name} no tiene créditos: déjalos en {music.with_suffix('.txt')} (o guarda la pista "
                         f"y su .txt con el mismo nombre en {library}). Sin ellos te pueden reclamar el vídeo.")
    lines = [line.rstrip() for line in own.read_text(encoding="utf-8").strip().splitlines()]
    if any("<<" in line for line in lines):
        raise SystemExit(f"Completa los créditos de la música en {own} (hay un hueco <<...>>)")
    return lines


def thumbnail_source(run_folder, work_folder, number=None):
    """montaje/<number> (01 by default: the hook) -> a still image path (a frame if it is a video,
    without the black bars some clips have burned in)."""
    def beat_of(p):
        return int(montage.NUMBER.match(p.stem).group(1))
    found = sorted((p for p in montage.folder_of(run_folder).iterdir()
                    if p.is_file() and manual.kind_of(p) and montage.NUMBER.match(p.stem)
                    and (number is None or beat_of(p) == number)),
                   key=lambda p: (beat_of(p), p.name))            # 1.mp4 before 02.mp4
    if not found:
        return None
    first = found[0]
    if manual.kind_of(first) == "video":
        frame = Path(work_folder) / "miniatura_original.jpg"
        bars = media.detect_bars(first, 0.5)
        media.ffmpeg("-ss", "0.5", "-i", first, *(["-vf", bars] if bars else []), "-frames:v", "1", "-q:v", "2",
                     frame)
        return frame
    return first


def thumbnail(image_path, text, font_path, dst, highlight="", side=False):
    """1280x720: the image filling the frame + the text, big, white with a black outline.
    side=False (shorts): centred at the bottom. side=True (long videos): on the half of the image
    with less detail, over a dark gradient, so it does not cover the subject; `highlight` (a word
    of the text) in yellow."""
    image = ImageOps.fit(Image.open(image_path).convert("RGB"), THUMB_SIZE, Image.Resampling.LANCZOS)
    if side:
        return side_thumbnail(image, text, font_path, dst, highlight)
    draw = ImageDraw.Draw(image)
    width, height = THUMB_SIZE
    words, size = text.upper().split(), round(height * 0.2)
    while True:                                                 # biggest size that fits in 2 lines
        font = ImageFont.truetype(str(font_path), size)
        lines = wrap(words, font, draw, width * 0.9)
        if len(lines) <= 2 or size < 40:
            break
        size -= 6
    stroke = max(4, size // 12)
    line_h = size * 1.1
    bottom = draw.textbbox((0, 0), "ÁGy", font=font, stroke_width=stroke)[3]    # the real height of a line
    y = height - height * 0.06 - bottom - (len(lines) - 1) * line_h
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((width - w) / 2, y), line, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
        y += line_h
    image.save(dst, quality=92)
    return dst


def quieter_side(image):
    """"left" or "right": the half with less detail (fewer edges), where the text covers less."""
    edges = image.convert("L").resize((128, 72)).filter(ImageFilter.FIND_EDGES)
    left = sum(edges.crop((0, 0, 64, 72)).getdata())
    right = sum(edges.crop((64, 0, 128, 72)).getdata())
    return "left" if left <= right else "right"


def _bare(word):
    return word.strip("¿?¡!.,:;«»\"'").upper()


def side_thumbnail(image, text, font_path, dst, highlight=""):
    width, height = THUMB_SIZE
    image = ImageEnhance.Color(ImageEnhance.Contrast(image).enhance(THUMB_CONTRAST)).enhance(THUMB_COLOR)
    side = quieter_side(image)
    # the dark gradient: strongest at the text's edge, gone past the middle
    shade_w = round(width * THUMB_SHADE)
    row = [round(THUMB_SHADE_ALPHA * (1 - x / shade_w) ** 1.5) for x in range(shade_w)]   # dark -> clear
    ramp = Image.new("L", (shade_w, 1))
    ramp.putdata(row if side == "left" else row[::-1])
    shade = Image.new("L", (width, height), 0)
    shade.paste(ramp.resize((shade_w, height)), (0 if side == "left" else width - shade_w, 0))
    image = Image.composite(Image.new("RGB", image.size, "black"), image, shade)

    draw = ImageDraw.Draw(image)
    lines, size = text_layout(text.upper().split(), font_path, draw, width * THUMB_TEXT_WIDTH, height * 0.8)
    font = ImageFont.truetype(str(font_path), size)
    stroke, line_h, margin = max(4, size // 12), size * 1.08, width * 0.05
    key = _bare(highlight) if highlight else ""
    y = (height - line_h * len(lines)) / 2
    space = draw.textlength(" ", font=font)
    for line in lines:
        w = draw.textlength(line, font=font)
        x = margin if side == "left" else width - margin - w
        for word in line.split():
            colour = THUMB_HIGHLIGHT if key and _bare(word) == key else "white"
            draw.text((x, y), word, font=font, fill=colour, stroke_width=stroke, stroke_fill="black")
            x += draw.textlength(word, font=font) + space
        y += line_h
    image.save(dst, quality=92)
    return dst


def splits(words, n):
    """Every way of cutting the words, in order, into n non-empty lines."""
    if n == 1:
        yield [" ".join(words)]
        return
    for cut in range(1, len(words) - n + 2):
        for rest in splits(words[cut:], n - 1):
            yield [" ".join(words[:cut])] + rest


def text_layout(words, font_path, draw, max_width, max_height, max_lines=3):
    """(lines, font size): of every way of putting the words in 1-3 lines, the one with the biggest
    letters. A short word (1-3 letters: "DE", "UN", "8") never takes a line alone, and fewer lines
    win when their letters are at least 75 % as big: a block of 1-2 lines reads at a glance."""
    reference = ImageFont.truetype(str(font_path), 100)
    def biggest(lines):
        widest = max(draw.textlength(l, font=reference) for l in lines)
        return min(100 * max_width / widest, max_height / (1.08 * len(lines)), THUMB_MAX_FONT)
    def lone_short(lines):
        return any(" " not in l and len(_bare(l)) <= 3 for l in lines) and len(lines) > 1
    best = []
    for n in range(1, min(max_lines, len(words)) + 1):
        options = [l for l in splits(words, n) if not lone_short(l)]
        if options:
            lines = max(options, key=biggest)
            best.append((lines, biggest(lines)))
    top = max(size for _, size in best)
    lines, size = next((l, s) for l, s in best if s >= top * 0.75)
    return lines, max(1, int(size))   # never wider than the box, even with an absurdly long word


def wrap(words, font, draw, max_width):
    lines, line = [], []
    for w in words:
        if line and draw.textlength(" ".join(line + [w]), font=font) > max_width:
            lines.append(" ".join(line))
            line = []
        line.append(w)
    return lines + ([" ".join(line)] if line else [])


# ------------------------------------------------------------------ prepare
def metadata_request(state, script):
    angle = state.get("angle") or {}
    parts = [f"Tema: {state.get('topic') or script.title}", f"Título del guion: {script.title}",
             f"Formato: {'short vertical (algo más de un minuto)' if state.get('format') == 'short' else 'vídeo largo'}"]
    if angle.get("title"):
        parts.append(f"Enfoque: {angle['title']}")
    if angle.get("question"):
        parts.append(f"Pregunta central: {angle['question']}")
    parts += ["", "# Guion"] + [s.narration for s in script.scenes]
    parts += ["", "# Datos en que se basa"] + [f"- {f['fact']}" for f in state.get("facts", [])]
    return "\n".join(parts)


def prepare(run_id, hub, thumb_number=None, force=False):
    settings = load_settings()
    options = settings.get("youtube", {})
    run_folder = runs_dir(run_id)
    state = run_state(run_id)
    script = Script.model_validate(state["script"])
    video = newest_video(run_folder)
    music = music_credits(run_folder, config.BASE / options.get("music_dir", "data/musica"))   # before paying the LLM
    folder = config.BASE / options.get("ready_dir", "data/listo_para_subir") / video.stem
    if folder.exists() and not force:
        raise SystemExit(f"{folder} already exists: edit it, or delete it (or use --force) to prepare it again")
    folder.mkdir(parents=True, exist_ok=True)

    draft = hub.chain(load_prompt(NODE), schema=PublishDraft, node=NODE).invoke(
        {"text": metadata_request(state, script)})
    starts = scene_starts(state, settings)
    synthetic = has_ai_content(run_folder, state)
    long_video = state.get("format") != "short"

    shutil.copy2(video, folder / "video.mp4")
    (folder / "descripcion.md").write_text(
        description(draft, chapters(state, starts) if long_video else [], state.get("sources", []),
                    credits(run_folder, state), synthetic, music) + "\n", encoding="utf-8")
    (folder / "subtitulos.srt").write_text(subtitles(state, starts), encoding="utf-8")
    source = thumbnail_source(run_folder, folder, thumb_number)
    thumb = thumbnail(source, draft.thumbnail_text, options.get("thumbnail_font", "C:/Windows/Fonts/ariblk.ttf"),
                      folder / "miniatura.jpg", highlight=draft.thumbnail_highlight, side=long_video) if source else None
    metadata = {
        "title": draft.title.strip()[:TITLE_MAX],
        "tags": draft.tags,
        "category_id": str(options.get("category_id", 28)),
        "default_language": options.get("default_language", "es"),
        "privacy": options.get("privacy", "private"),
        "publish_at": None,               # "2026-10-12T18:00:00+02:00": scheduled (private until then)
        "made_for_kids": options.get("made_for_kids", False),
        "synthetic_media": synthetic,
        "license": "youtube",             # youtube | creativeCommon
        "video": "video.mp4",
        "description": "descripcion.md",
        "thumbnail": thumb.name if thumb else None,
        "thumbnail_text": draft.thumbnail_text,
        "thumbnail_highlight": draft.thumbnail_highlight,
        "subtitles": "subtitulos.srt",
        "run_id": run_id,
        "source_video": str(video),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder, metadata
