"""ffmpeg helpers. The ffmpeg used is the one inside the venv (imageio-ffmpeg):
nothing to install on the system, and it has h264_nvenc, loudnorm, xfade and ass.
"""
import json
import os
import re
import subprocess
import wave
from functools import cache
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 48_000


def ffmpeg(*args, cwd=None):
    """Runs ffmpeg and returns its stderr (where ffmpeg prints its reports). Raises with the tail of the log."""
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-y", *map(str, args)], cwd=cwd,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{result.stderr[-1500:]}")
    return result.stderr


def ffmpeg_frames(frames, width, height, fps, *output_args):
    """Encodes frames made in Python (PIL images or raw RGB bytes, width x height) with ffmpeg:
    for motion ffmpeg's filters cannot make smooth (zoompan moves in whole pixels and jitters)."""
    process = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                                "-i", "-", *map(str, output_args)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame if isinstance(frame, bytes) else frame.tobytes())
    except BrokenPipeError:
        pass                                            # ffmpeg stopped: its error is read below
    finally:
        process.stdin.close()
    error = process.stderr.read().decode("utf-8", "replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed ({process.returncode}):\n{error[-1500:]}")


def media_duration(path):
    """Seconds of any audio/video file, read from ffmpeg's report (imageio-ffmpeg has no ffprobe)."""
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", str(path)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    found = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not found:
        raise RuntimeError(f"Cannot read the duration of {path}:\n{result.stderr[-500:]}")
    h, m, s = found.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def detect_bars(path, start=0.0, seconds=2.0, min_share=0.02):
    """Black bars burned into a clip (letterbox/pillarbox) -> 'crop=w:h:x:y' to remove them,
    or None if there are none (less than min_share of the width or height)."""
    report = ffmpeg("-ss", f"{start:.3f}", "-i", path, "-t", seconds, "-vf", "cropdetect=limit=24:round=2:reset=0",
                    "-an", "-f", "null", "-")
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", report)
    size = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", report)
    if not found or not size:
        return None
    w, h, x, y = map(int, found[-1])
    full_w, full_h = map(int, size.groups())
    if w <= 0 or h <= 0:
        return None
    # Real bars are the same on both sides; a dark part of the scene next to one is not a bar.
    # So only the smaller side is removed, on both sides (even: yuv420p needs even sizes).
    bar_x = min(x, full_w - w - x) // 2 * 2
    bar_y = min(y, full_h - h - y) // 2 * 2
    if bar_x < full_w * min_share and bar_y < full_h * min_share:
        return None
    return f"crop={full_w - 2 * bar_x}:{full_h - 2 * bar_y}:{bar_x}:{bar_y}"


def filter_path(path, cwd):
    """A path to write INSIDE a filtergraph (ass=..., fontsdir=...): relative to cwd and with '/',
    because a Windows drive's ':' breaks the filter syntax. On another drive: escaped absolute."""
    try:
        return Path(os.path.relpath(path, cwd)).as_posix()
    except ValueError:
        return "'" + Path(path).resolve().as_posix().replace(":", r"\:") + "'"


# ------------------------------------------------------------------ video encoding
@cache
def video_encoder(preferred="auto"):
    """h264_nvenc if the GPU can use it (checked once), otherwise libx264."""
    if preferred != "auto":
        return preferred
    try:
        ffmpeg("-f", "lavfi", "-i", "color=s=256x256:d=0.1", "-frames:v", 1, "-c:v", "h264_nvenc", "-f", "null", "-")
        return "h264_nvenc"
    except RuntimeError:
        return "libx264"


def encode_args(quality, fps, encoder="auto"):
    """H.264 arguments. quality: constant quality (lower = better; ~16 intermediate, ~21 final)."""
    encoder = video_encoder(encoder)
    common = ["-pix_fmt", "yuv420p", "-r", fps, "-g", fps * 2]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", quality,
                "-b:v", 0, "-profile:v", "high", *common]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", quality, "-profile:v", "high", *common]


def concat_videos(paths, dst):
    """Joins clips with the SAME codec and settings, without re-encoding (concat demuxer)."""
    dst = Path(dst)
    listing = dst.with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    ffmpeg("-f", "concat", "-safe", 0, "-i", listing, "-c", "copy", dst)
    listing.unlink()
    return dst


def concat_wavs(paths, lengths, dst):
    """Joins mono 16-bit WAVs, each padded with silence to its length (seconds), so every
    piece starts exactly where its video segment starts. All must share the sample rate."""
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        written = 0.0
        for path, length in zip(paths, lengths):
            with wave.open(str(path)) as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
                    raise ValueError(f"{path}: expected mono 16-bit {SAMPLE_RATE} Hz")
                samples = w.readframes(w.getnframes())
            written += length                     # cumulative: no rounding drift along the video
            target = round(written * SAMPLE_RATE) - out.tell()   # tell(): frames written so far
            samples = samples[:target * 2]
            out.writeframes(samples + b"\0\0" * (target - len(samples) // 2))
    return dst


# Removes silence at the start and at the end (the end: reversing, trimming the start, reversing
# back), keeping `keep` seconds of it so the speech does not sound clipped.
def _trim_silence(keep_start=0.05, keep_end=0.2, threshold="-45dB"):
    trim = "silenceremove=start_periods=1:start_silence={}:start_threshold=" + threshold
    return f"{trim.format(keep_start)},areverse,{trim.format(keep_end)},areverse,"


def normalize_loudness(src, dst, lufs=-14, true_peak=-1.5, lra=11, end=None, trim_silence=False):
    """Any audio -> mono WAV at `lufs` (EBU R128), in two passes: the first measures, the
    second applies the gain LINEARLY (same length and timing, no "pumping" like one pass).
    end: cut the audio at this second (when the word times are known, e.g. edge-tts).
    trim_silence: remove the silence at the start and end by level (when they are not known).
    The trim is a pass of its own: chained with loudnorm in one filter graph, ffmpeg 7.1 aborts
    ("Assertion best_input >= 0 failed") on some audios, e.g. one with no silence at either end."""
    cut = ["-t", round(end, 3)] if end else []
    trimmed = None
    if trim_silence:
        trimmed = Path(dst).with_name(Path(dst).stem + ".trim.wav")
        ffmpeg(*cut, "-i", src, "-af", _trim_silence().rstrip(","), "-c:a", "pcm_s16le", trimmed)
        src, cut = trimmed, []
    target = f"I={lufs}:TP={true_peak}:LRA={lra}"
    report = ffmpeg(*cut, "-i", src, "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-")
    measured = json.loads(re.findall(r"\{[^{}]*\}", report)[-1])
    ffmpeg(*cut, "-i", src, "-af",
           f"loudnorm={target}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
           f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
           f":offset={measured['target_offset']}:linear=true",
           "-ar", SAMPLE_RATE, "-ac", 1, "-c:a", "pcm_s16le", dst)     # loudnorm outputs 192 kHz otherwise
    if trimmed:
        trimmed.unlink()
    return dst


def loudness(src):
    """Integrated loudness (LUFS) of an audio file."""
    report = ffmpeg("-i", src, "-af", "loudnorm=print_format=json", "-f", "null", "-")
    return float(json.loads(re.findall(r"\{[^{}]*\}", report)[-1])["input_i"])


def music_bed(src, seconds, dst, lufs, eq_db=-4, fade_in=2.0, fade_out=3.0):
    """Your music as the background of a video `seconds` long: looped (or cut) to that length,
    a dip of eq_db around 1-4 kHz (the voice's frequencies: it is heard but does not cover the
    words), fades at both ends, and a FIXED gain to `lufs`: it never goes up or down with the voice."""
    shaped = Path(dst).with_name(Path(dst).stem + ".shaped.wav")
    fade_out = min(fade_out, seconds / 2)
    ffmpeg("-stream_loop", -1, "-i", src, "-t", round(seconds, 3), "-af",
           f"equalizer=f=2000:t=o:w=2:g={eq_db},afade=t=in:d={fade_in},"
           f"afade=t=out:st={seconds - fade_out:.3f}:d={fade_out}",
           "-ar", SAMPLE_RATE, "-ac", 1, "-c:a", "pcm_s16le", shaped)
    gain = lufs - loudness(shaped)
    ffmpeg("-i", shaped, "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_s16le", dst)
    shaped.unlink()
    return Path(dst)


def mix_music(voice, bed, dst, lufs):
    """Voice + music bed (both mono WAVs at SAMPLE_RATE) -> dst, normalized again to `lufs` (the
    same linear two-pass loudnorm as the narration: no pumping). Replaces the voice's file."""
    mixed = Path(dst).with_name(Path(dst).stem + ".raw.wav")
    ffmpeg("-i", voice, "-i", bed, "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0:duration=first",
           "-ar", SAMPLE_RATE, "-ac", 1, "-c:a", "pcm_s16le", mixed)
    normalize_loudness(mixed, dst, lufs)
    mixed.unlink()
    Path(voice).unlink()
    return Path(dst)


def wav_duration(path):
    """Seconds of a PCM WAV (no ffprobe needed: imageio-ffmpeg does not include it)."""
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()
