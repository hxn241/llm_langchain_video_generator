"""assemble: visuals + narration (+ subtitles, music) -> <run>/<yyyymmdd>_<topic>.mp4 (the date it is
assembled; the topic without accents or symbols, e.g. 20261011_Google_lanza_su_primer_satelite.mp4).

1. Each scene lasts EXACTLY its audio + video.scene_gap, rounded to whole frames, and its
   narration starts where the scene starts. Inside it, one video segment per BEAT: from
   the beat's first word (graph/nodes/beats.py, times from align) to the next beat, so
   each image appears right when what it shows is said. The files are those of <run>/montaje/
   as you left them at the review_visuals pause (graph/montage.py): a beat without a file
   lengthens the previous shot, one with several shares its time among them.
   - video: black bars burned into the clip are removed, then it is cropped to fill the
     frame. If it is shorter than the beat it is slowed down
     (at most MAX_SLOWDOWN) and its last frame is held for the rest.
   - photo / generated image: a slow, subtle camera move (MOTIONS: zoom to the centre or to a
     side, pans, tilts; a different one for each photo), made frame by frame at sub-pixel
     positions (ffmpeg's zoompan jitters) and piped to ffmpeg. With video.parallax, 2.5D:
     a GPU worker (tts_worker/parallax_worker.py) estimates the photo's depth and moves the
     foreground a little more than the background; if it fails, the plain move (ken_burns).
   - none:  a plain dark background.
   Runs made before beats existed: one beat per scene (visuals keyed by scene id).
   Transitions (video.transition): a crossfade between SCENES only (inside a scene the cut on
   the key word stays). The last shot of a scene is made longer by the fade and overlaps the
   next one, so every scene still starts exactly where its narration starts.
2. The segments are joined without re-encoding; the scene WAVs are joined padded with
   silence to the same lengths (no drift: positions are cumulative) and the whole
   narration is normalized again to video.loudness_lufs. Music (montaje/musica.mp3, yours):
   looped or cut to the video, at a FIXED level video.music.db under the voice (no ducking),
   with a dip in the voice's frequencies and fades; the mix is normalized again.
3. Final pass: narration + subtitles (ASS, graph/subtitles.py; video.subtitles) + the labels
   (rótulos: source, person, key term; <run>/rotulos.json -> graph/onscreen.py) burned in,
   h264_nvenc, AAC.

Segments stay in <run>/segments/ (to redo one scene later).
"""
import json
import os
import re
import unicodedata
from datetime import date
from pathlib import Path

from PIL import Image, ImageOps

from graph import montage, onscreen, subtitles
from graph.nodes import labels as labels_node
from graph.state import Script
from llmkit.config import BASE
from services import media, workers
from services.settings import load_settings, runs_dir

MAX_SLOWDOWN = 1.5          # a short clip may play this much slower before its last frame is held
ZOOM = 0.10                 # Ken Burns: the photo is made 10 % bigger than the frame: room to move
# Camera moves of the photos (subtle; a different one each photo, in this order):
# zoom z0 -> z1 (1 = the whole enlarged photo, 1 + ZOOM = the frame at 1:1) and the centre from
# (x0, y0) to (x1, y1), in [-1, 1] of the room left at that zoom (-1 left/top, 1 right/bottom).
MOTIONS = [
    {"name": "zoom in", "z0": 1.0, "z1": 1 + ZOOM, "x0": 0, "y0": 0, "x1": 0, "y1": 0},
    {"name": "pan right", "z0": 1.06, "z1": 1.06, "x0": -0.8, "y0": 0, "x1": 0.8, "y1": 0},
    {"name": "zoom out", "z0": 1 + ZOOM, "z1": 1.0, "x0": 0, "y0": 0, "x1": 0, "y1": 0},
    {"name": "zoom in left", "z0": 1.0, "z1": 1 + ZOOM, "x0": 0, "y0": 0, "x1": -0.7, "y1": -0.3},
    {"name": "tilt up", "z0": 1.06, "z1": 1.06, "x0": 0, "y0": 0.8, "x1": 0, "y1": -0.8},
    {"name": "zoom in right", "z0": 1.0, "z1": 1 + ZOOM, "x0": 0, "y0": 0, "x1": 0.7, "y1": -0.3},
    {"name": "pan left", "z0": 1.06, "z1": 1.06, "x0": 0.8, "y0": 0, "x1": -0.8, "y1": 0},
    {"name": "zoom out up", "z0": 1 + ZOOM, "z1": 1.02, "x0": 0, "y0": -0.6, "x1": 0, "y1": 0},
]
BACKGROUND = "0x14181F"     # kind "none"
SEGMENT_QUALITY = 16        # intermediate segments: near lossless (they are encoded twice)
MAX_NAME_CHARS = 80         # the topic part of the video's file name


def video_name(topic, day=None):
    """'Google lanza su primer satélite...' -> '20261011_Google_lanza_su_primer_satelite....mp4':
    the date, then the topic without accents and with _ for anything that is not a letter or digit."""
    plain = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", plain).strip("_")[:MAX_NAME_CHARS].rstrip("_") or "video"
    return f"{(day or date.today()):%Y%m%d}_{slug}.mp4"


def scene_frames(audio, scene_ids, fps, gap):
    """Frames of each scene: audio + gap, rounded to whole frames."""
    return {i: max(1, round((audio[i]["duration"] + gap) * fps)) for i in scene_ids}


def beat_frames(beats, scene_ids, frames, fps):
    """[(beat, frames)] in order: each beat from its start to the next beat of its scene
    (or the scene's end). Without beats: one per scene, with the scene's id."""
    by_scene = {}
    for b in beats or []:
        by_scene.setdefault(b["scene"], []).append(b)
    out = []
    for sid in scene_ids:
        scene_beats = sorted(by_scene.get(sid) or [{"id": sid, "scene": sid, "start": 0.0}], key=lambda b: b["start"])
        cuts = [0] + [min(frames[sid], round(b["start"] * fps)) for b in scene_beats[1:]]
        ends = cuts[1:] + [frames[sid]]
        out += [(b, end - cut) for b, cut, end in zip(scene_beats, cuts, ends) if end > cut]
    return out


def shots(timeline, visuals, montage=None):
    """[(beat, frames)] -> [(beat, visual, frames)], the segments to make, in order.
    With `montage` (review_visuals: beat id -> its files in <run>/montaje/): a beat with no file
    lengthens the previous shot (or the next one, at the start); one with several shares its
    frames among them. Without it (runs from before montaje/): the visual chosen for each beat."""
    out, pending = [], 0
    for beat, count in timeline:
        files = (montage.get(beat["id"], []) if montage is not None
                 else [visuals.get(beat["id"]) or {"kind": "none", "path": ""}])
        if not files:
            if out:
                b, v, f = out[-1]
                out[-1] = (b, v, f + count)
            else:
                pending += count
            continue
        count, pending = count + pending, 0
        cuts = [round(count * k / len(files)) for k in range(len(files) + 1)]
        out += [(beat, visual, cuts[k + 1] - cuts[k]) for k, visual in enumerate(files) if cuts[k + 1] > cuts[k]]
    if pending:                                  # every file removed: a dark background
        out.append((timeline[-1][0], {"kind": "none", "path": ""}, pending))
    return out


def fade_frames(timeline, fade):
    """Extra frames of each shot: `fade` for the last shot of a scene (it overlaps the next scene's
    first shot), 0 for the rest. No fade where either shot is not longer than the fade itself."""
    extra = [0] * len(timeline)
    for n in range(len(timeline) - 1):
        (a, _, a_frames), (b, _, b_frames) = timeline[n], timeline[n + 1]
        if fade and a.get("scene", a["id"]) != b.get("scene", b["id"]) and a_frames > fade and b_frames > fade:
            extra[n] = fade
    return extra


def crossfade_scenes(paths, counts, extra, fade, fps, encoder, folder):
    """The segments to join: shots linked by a fade (extra[n] > 0: n fades into n + 1) become one
    file with ffmpeg's xfade. A fade starts where the next shot's time starts, so the joined video
    has exactly sum(counts) frames: no shift against the narration."""
    out, group = [], []
    for n, path in enumerate(paths):
        group.append(n)
        if extra[n]:
            continue
        if len(group) == 1:
            out.append(path)
        else:
            joined = folder / f"scene_join_{group[0] + 1:03d}.mp4"
            inputs, chain, label, offset = [], [], "0:v", 0
            for k, m in enumerate(group):
                inputs += ["-i", paths[m]]
                if k:
                    offset += counts[group[k - 1]]
                    chain.append(f"[{label}][{k}:v]xfade=transition=fade:duration={fade / fps:.4f}"
                                 f":offset={offset / fps:.4f}[x{k}]")
                    label = f"x{k}"
            media.ffmpeg(*inputs, "-filter_complex", ";".join(chain), "-map", f"[{label}]",
                         "-frames:v", sum(counts[m] for m in group),
                         *media.encode_args(SEGMENT_QUALITY, fps, encoder), joined)
            out.append(joined)
        group = []
    return out


def segment_args(visual, width, height, fps, frames):
    """ffmpeg input + filter arguments for a video or empty segment (without the output part).
    Photos are made frame by frame instead (ken_burns)."""
    seconds = frames / fps
    if visual["kind"] == "video":
        clip = media.media_duration(visual["path"])
        slow = min(MAX_SLOWDOWN, seconds / clip) if clip < seconds else 1.0
        start = min(0.5, clip - seconds) if clip > seconds + 0.5 else 0      # skip a possible fade-in
        bars = media.detect_bars(visual["path"], start + min(1.0, clip / 4))  # letterbox burned into the clip
        vf = ((bars + "," if bars else "") +
              f"setpts={slow:.4f}*PTS,fps={fps},"
              f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={width}:{height},setsar=1,tpad=stop_mode=clone:stop_duration={seconds:.3f}")
        return ["-ss", f"{start:.3f}", "-i", visual["path"], "-vf", vf, "-an"]
    return ["-f", "lavfi", "-i", f"color=c={BACKGROUND}:s={width}x{height}:r={fps}"]


def camera(motion, big_w, big_h, t):
    """Where the frame is cut from the enlarged photo at t (0..1): (centre x, centre y, w, h) in its
    pixels. The same maths in tts_worker/parallax_worker.py."""
    e = t * t * (3 - 2 * t)                              # smoothstep: starts and ends gently
    zoom = motion["z0"] + (motion["z1"] - motion["z0"]) * e
    w, h = big_w / zoom, big_h / zoom
    x = motion["x0"] + (motion["x1"] - motion["x0"]) * e
    y = motion["y0"] + (motion["y1"] - motion["y0"]) * e
    return big_w / 2 + x * (big_w - w) / 2, big_h / 2 + y * (big_h - h) / 2, w, h


def ken_burns(path, width, height, frames, motion):
    """The frames of a slow camera move on a photo (PIL images). Each frame is cut at sub-pixel
    positions and interpolated, so the move is smooth (ffmpeg's zoompan rounds to whole pixels: it
    jitters). The photo is first resized (Lanczos) to the frame plus ZOOM, so every frame is scaled
    by less than 1 + ZOOM: no shimmer from shrinking a big image without filtering."""
    image = ImageOps.fit(Image.open(path).convert("RGB"),
                         (round(width * (1 + ZOOM)), round(height * (1 + ZOOM))), Image.Resampling.LANCZOS)
    big_w, big_h = image.size
    for n in range(frames):
        cx, cy, w, h = camera(motion, big_w, big_h, n / max(1, frames - 1))
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        yield image.transform((width, height), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)


def parallax_segments(jobs, width, height, fps, encoder, settings):
    """2.5D for the photos, all in one call to the GPU worker (the depth model loads once).
    jobs: [{"image", "out", "frames", "motion"}] -> the set of outs it made; empty if it failed
    (those photos get the plain move)."""
    options = settings["video"].get("parallax") or {}
    if not jobs or not options.get("enabled"):
        return set()
    try:
        result = workers.run("parallax", {
            "model": options.get("model", "depth-anything/Depth-Anything-V2-Small-hf"),
            "strength": options.get("strength", 0.02), "zoom": ZOOM, "ffmpeg": media.FFMPEG,
            "width": width, "height": height, "fps": fps,
            "encode": [str(a) for a in media.encode_args(SEGMENT_QUALITY, fps, encoder)], "items": jobs}, settings)
    except Exception as e:                                   # noqa: BLE001 - the plain move is fine too
        print(f"  ⚠️  assemble: sin 2.5D ({str(e).strip().splitlines()[-1][:200]}): movimiento normal")
        return set()
    flat = sum(1 for item in result["items"] if item.get("flat"))
    print(f"  · assemble: 2.5D en {len(result['items'])} fotos ({result['seconds']} s"
          + (f"; {flat} planas, sin profundidad" if flat else "") + ")")
    return {item["out"] for item in result["items"] if item.get("ok")}


def fonts_dir(settings):
    """Where libass looks for video.font: assets/fonts if it has fonts, otherwise the system's
    (Windows: C:/Windows/Fonts; elsewhere None = fontconfig finds them)."""
    folder = BASE / settings["video"].get("fonts_dir", "assets/fonts")
    if folder.is_dir() and any(folder.iterdir()):
        return folder
    system = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    return system if os.name == "nt" and system.is_dir() else None


def ass_filter(path, settings):
    fonts = fonts_dir(settings)
    return f"ass={media.filter_path(path, BASE)}" + (f":fontsdir={media.filter_path(fonts, BASE)}" if fonts else "")


def labels_now(folder, script, audio, labels):
    """The labels to draw: <run>/rotulos.json if it is there (your edits count), on the current times;
    otherwise the state's."""
    path = folder / labels_node.FILE
    if not path.exists():
        return labels or []
    saved = json.loads(path.read_text(encoding="utf-8"))
    return [p for p in (labels_node.place(l, script, audio) for l in saved) if p]


def without_photo(label):
    return {k: v for k, v in label.items() if not k.startswith("foto")}


def photo_overlays(photos, folder, work, fps, filters):
    """The labels' portraits over the video (onscreen.photos): each one framed in white at its size
    (cut to 4:5, the face is usually in the upper part), slid in and faded like the text, and the ASS
    filters after them (the text on top). -> (extra ffmpeg inputs, filter_complex) or ([], None)."""
    if not photos:
        return [], None
    inputs, chain, last = [], [], "0:v"
    for n, p in enumerate(photos):
        border = max(2, round(p["h"] * 0.015))
        with Image.open(folder / p["path"]) as image:
            framed = ImageOps.expand(ImageOps.fit(image.convert("RGB"), (p["w"] - 2 * border, p["h"] - 2 * border),
                                                  Image.Resampling.LANCZOS, centering=(0.5, 0.3)),
                                     border=border, fill="white")
        still = work / f"retrato_{n + 1:02d}.png"
        framed.save(still)
        inputs += ["-loop", 1, "-framerate", fps, "-t", f"{p['end']:.3f}", "-i", still]
        s, e = p["start"], p["end"]
        chain.append(f"[{n + 2}:v]format=rgba,fade=t=in:st={s:.3f}:d={onscreen.FADE_IN}:alpha=1,"
                     f"fade=t=out:st={e - onscreen.FADE_OUT:.3f}:d={onscreen.FADE_OUT}:alpha=1[p{n}]")
        slide = f"{p['x']}-{onscreen.SLIDE}*max(0,1-(t-{s:.3f})/{onscreen.SLIDE_SECONDS})"
        chain.append(f"[{last}][p{n}]overlay=x='{slide}':y={p['y']}:enable='between(t,{s:.3f},{e:.3f})'"
                     f":eof_action=pass[o{n}]")
        last = f"o{n}"
    chain.append(f"[{last}]{','.join(filters) or 'null'}[v]")
    return inputs, ";".join(chain)


def make_assemble(settings=None):
    settings = settings or load_settings()

    def assemble(state, config):
        video = settings["video"]
        video_format = state.get("format") or settings.get("format", "long")
        width, height, fps = video[video_format]["width"], video[video_format]["height"], video["fps"]
        encoder = video.get("encoder", "auto")
        script = Script.model_validate(state["script"])
        audio, visuals = state["audio"], state["visuals"]
        folder = runs_dir(state.get("run_id") or config["configurable"]["thread_id"])
        segments_dir = folder / "segments"
        segments_dir.mkdir(exist_ok=True)

        ids = [s.id for s in script.scenes]
        frames = scene_frames(audio, ids, fps, video.get("scene_gap", 0.15))
        lengths = {i: frames[i] / fps for i in ids}
        starts, t = {}, 0.0
        for i in ids:
            starts[i], t = t, t + lengths[i]

        # 1. one segment per beat (the last one of a scene longer by the fade into the next)
        timeline = shots(beat_frames(state.get("beats"), ids, frames, fps), visuals, state.get("montage"))
        fade = round((video.get("transition") or {}).get(video_format, 0) * fps)
        extra = fade_frames(timeline, fade)
        for old in segments_dir.glob("*.mp4"):          # names change when files are removed or added
            old.unlink()
        paths = [segments_dir / f"shot_{n:03d}.mp4" for n in range(1, len(timeline) + 1)]
        photos = [n for n, (_, visual, _) in enumerate(timeline) if visual["kind"] in ("photo", "generated")]
        motion = {n: MOTIONS[k % len(MOTIONS)] for k, n in enumerate(photos)}    # a different move each photo
        done = parallax_segments([{"image": timeline[n][1]["path"], "out": str(paths[n]),
                                   "frames": timeline[n][2] + extra[n], "motion": motion[n]} for n in photos],
                                 width, height, fps, encoder, settings)
        for n, (beat, visual, count) in enumerate(timeline):
            count += extra[n]
            output = ["-frames:v", count, *media.encode_args(SEGMENT_QUALITY, fps, encoder), paths[n]]
            if str(paths[n]) in done:
                pass                                     # 2.5D, already made by the worker
            elif n in motion:
                media.ffmpeg_frames(ken_burns(visual["path"], width, height, count, motion[n]),
                                    width, height, fps, *output)
            else:
                media.ffmpeg(*segment_args(visual, width, height, fps, count), *output)
            print(f"  · assemble: {n + 1}/{len(timeline)} fragmento {beat['id']} · "
                  f"{Path(visual['path']).name or 'fondo'} ({visual['kind']}, {count / fps:.1f} s"
                  + (f", {motion[n]['name']}{' 2.5D' if str(paths[n]) in done else ''}" if n in motion else "") + ")")
        segments = crossfade_scenes(paths, [c for _, _, c in timeline], extra, fade, fps, encoder, segments_dir)

        # 2. joined video and narration (+ your music)
        joined = media.concat_videos(segments, folder / "video_only.mp4")
        joined_wav = media.concat_wavs([audio[i]["path"] for i in ids], [lengths[i] for i in ids],
                                       folder / "narration_raw.wav")
        # each scene is already at the target, but the pauses lower the whole: measured and set again
        narration = media.normalize_loudness(joined_wav, folder / "narration.wav", video["loudness_lufs"])
        joined_wav.unlink()
        music = montage.music_of(folder)
        if music:
            options = video.get("music") or {}
            bed = media.music_bed(music, sum(lengths.values()), folder / "music_bed.wav",
                                  video["loudness_lufs"] + options.get("db", -20), options.get("eq_db", -4),
                                  options.get("fade_in", 2), options.get("fade_out", 3))
            narration = media.mix_music(narration, bed, folder / "narration_music.wav", video["loudness_lufs"])
            bed.unlink()
            print(f"  · assemble: música {music.name} a {options.get('db', -20)} dB bajo la voz")

        # 3. final pass: subtitles burned in (if any) + audio
        burned = subtitles.burned_document(audio, starts, lengths, width, height, video.get("font", "Arial"),
                                           (video.get("subtitles") or {}).get(video_format) or {})
        subtitles_path = folder / "subtitulos.ass"
        if burned:
            subtitles_path.write_text(burned, encoding="utf-8")
        # the labels (rótulos): rotulos.json as it is now (you may have edited it at the pause)
        label_options = video.get("labels") or {}
        labels = (labels_now(folder, script, audio, state.get("labels"))
                  if label_options.get("enabled", True) else [])
        labels = [l if not l.get("foto") or (folder / l["foto"]).is_file() else without_photo(l) for l in labels]
        final = folder / video_name(state.get("topic") or script.title)

        def final_pass(labels):
            filters = [ass_filter(subtitles_path, settings)] if burned else []
            on_screen = onscreen.document(labels, starts, lengths, width, height, video.get("font", "Arial"),
                                          label_options)
            labels_path = folder / "rotulos.ass"
            if on_screen:
                labels_path.write_text(on_screen, encoding="utf-8")
                filters.append(ass_filter(labels_path, settings))
            inputs, graph = photo_overlays(onscreen.photos(labels, starts, lengths, width, height, label_options),
                                           folder, segments_dir, fps, filters)
            if graph:
                video_args = ["-filter_complex", graph, "-map", "[v]"]
            else:
                video_args = ["-map", "0:v"] + (["-vf", ",".join(filters)] if filters else [])
            video_args += media.encode_args(video.get("quality", 21), fps, encoder) if filters or graph else ["-c:v", "copy"]
            media.ffmpeg("-i", joined, "-i", narration, *inputs, *video_args, "-map", "1:a",
                         "-c:a", "aac", "-b:a", "192k", "-ar", media.SAMPLE_RATE, "-movflags", "+faststart",
                         final, cwd=BASE)

        try:
            final_pass(labels)
        except Exception as e:                       # noqa: BLE001 - without the photos it is the video as before
            if not any(l.get("foto") for l in labels):
                raise
            print(f"  ⚠️  assemble: los retratos fallaron ({str(e).strip().splitlines()[-1][:200]}): "
                  "los rótulos van solo con texto")
            final_pass([without_photo(l) for l in labels])
        joined.unlink()
        return {"video_path": str(final), **({"subtitles_path": str(subtitles_path)} if burned else {})}

    return assemble
