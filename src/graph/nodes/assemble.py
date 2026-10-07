"""assemble: visuals + narration (+ on-screen text) -> <run>/final.mp4.

1. Each scene lasts EXACTLY its audio + video.scene_gap, rounded to whole frames, and its
   narration starts where the scene starts. Inside it, one video segment per BEAT: from
   the beat's first word (graph/nodes/beats.py, times from align) to the next beat, so
   each image appears right when what it shows is said.
   - video: black bars burned into the clip are removed, then it is cropped to fill the
     frame. If it is shorter than the beat it is slowed down
     (at most MAX_SLOWDOWN) and its last frame is held for the rest.
   - photo / generated image: slow zoom (Ken Burns), alternating in and out, made frame by
     frame in Python at sub-pixel positions (ffmpeg's zoompan jitters) and piped to ffmpeg.
   - none:  a plain dark background.
   Runs made before beats existed: one beat per scene (visuals keyed by scene id).
2. The segments are joined without re-encoding; the scene WAVs are joined padded with
   silence to the same lengths (no drift: positions are cumulative) and the whole
   narration is normalized again to video.loudness_lufs.
3. Final pass: narration + on-screen text (ASS) burned in, h264_nvenc, AAC.

Segments stay in <run>/segments/ (to redo one scene later). Next (README): transitions,
music with ducking, word-by-word subtitles (the subtitles node will set subtitles_path).
"""
import os
from pathlib import Path

from PIL import Image, ImageOps

from graph.state import Script
from llmkit.config import BASE
from services import ass, media
from services.settings import load_settings, runs_dir

MAX_SLOWDOWN = 1.5          # a short clip may play this much slower before its last frame is held
ZOOM = 0.10                 # Ken Burns: the photo grows (or shrinks) 10 % over the beat
BACKGROUND = "0x14181F"     # kind "none"
SEGMENT_QUALITY = 16        # intermediate segments: near lossless (they are encoded twice)


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


def ken_burns(path, width, height, frames, zoom_in=True):
    """The frames of a slow zoom on a photo (PIL images). Each frame is cut at sub-pixel positions
    and interpolated, so the zoom is smooth (ffmpeg's zoompan rounds to whole pixels: it jitters).
    The photo is first resized (Lanczos) to the frame plus the zoom, so every frame is scaled by
    less than 1 + ZOOM: no shimmer from shrinking a big image without filtering."""
    image = ImageOps.fit(Image.open(path).convert("RGB"),
                         (round(width * (1 + ZOOM)), round(height * (1 + ZOOM))), Image.Resampling.LANCZOS)
    big_w, big_h = image.size
    for n in range(frames):
        t = n / max(1, frames - 1)
        t = t * t * (3 - 2 * t)                          # smoothstep: starts and ends gently
        zoom = 1 + ZOOM * (t if zoom_in else 1 - t)       # 1 = the whole image, 1 + ZOOM = the frame at 1:1
        w, h = big_w / zoom, big_h / zoom
        box = ((big_w - w) / 2, (big_h - h) / 2, (big_w + w) / 2, (big_h + h) / 2)
        yield image.transform((width, height), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)


def overlay_document(script, starts, lengths, width, height, font):
    """ASS with each scene's on_screen_text, at the top, during its whole scene. None if there is none."""
    events = [ass.event(starts[s.id] + 0.2, starts[s.id] + lengths[s.id] - 0.1, "OnScreen", s.on_screen_text)
              for s in script.scenes if s.on_screen_text]
    if not events:
        return None
    portrait = height > width
    size = round(min(width, height) * 0.075)
    styles = [ass.style("OnScreen", font, size, outline_colour=ass.colour("000000", 0x60), box=True,
                        outline=round(size * 0.25), alignment=ass.TOP, margin_h=round(width * 0.08),
                        margin_v=round(height * (0.14 if portrait else 0.08)))]
    return ass.document(width, height, styles, events)


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

        # 1. one segment per beat
        segments = []
        timeline = beat_frames(state.get("beats"), ids, frames, fps)
        for n, (beat, count) in enumerate(timeline, start=1):
            visual = visuals.get(beat["id"]) or {"kind": "none", "path": ""}
            path = segments_dir / f"beat_{beat['id']:03d}.mp4"
            output = ["-frames:v", count, *media.encode_args(SEGMENT_QUALITY, fps, encoder), path]
            if visual["kind"] in ("photo", "generated"):
                media.ffmpeg_frames(ken_burns(visual["path"], width, height, count, zoom_in=n % 2 == 1),
                                    width, height, fps, *output)
            else:
                media.ffmpeg(*segment_args(visual, width, height, fps, count), *output)
            segments.append(path)
            print(f"  · assemble: {n}/{len(timeline)} escena {beat['scene']} ({visual['kind']}, {count / fps:.1f} s)")

        # 2. joined video and narration
        joined = media.concat_videos(segments, folder / "video_only.mp4")
        joined_wav = media.concat_wavs([audio[i]["path"] for i in ids], [lengths[i] for i in ids],
                                       folder / "narration_raw.wav")
        # each scene is already at the target, but the pauses lower the whole: measured and set again
        narration = media.normalize_loudness(joined_wav, folder / "narration.wav", video["loudness_lufs"])
        joined_wav.unlink()

        # 3. final pass: text burned in (if any) + audio
        filters = []
        overlay = overlay_document(script, starts, lengths, width, height, video.get("font", "Arial"))
        if overlay:
            (folder / "overlay.ass").write_text(overlay, encoding="utf-8")
            filters.append(ass_filter(folder / "overlay.ass", settings))
        if state.get("subtitles_path"):
            filters.append(ass_filter(state["subtitles_path"], settings))
        video_args = (["-vf", ",".join(filters), *media.encode_args(video.get("quality", 21), fps, encoder)]
                      if filters else ["-c:v", "copy"])
        final = folder / "final.mp4"
        media.ffmpeg("-i", joined, "-i", narration, "-map", "0:v", "-map", "1:a", *video_args,
                     "-c:a", "aac", "-b:a", "192k", "-ar", media.SAMPLE_RATE, "-movflags", "+faststart",
                     final, cwd=BASE)
        joined.unlink()
        return {"video_path": str(final)}

    return assemble
