"""Text to speech.

Qwen3-TTS (voice.mode=clone): local, Apache 2.0, on the GPU. It runs in its own
environment (.venv-tts, with torch) through src/tts_worker/qwen_worker.py: each
call loads the model, generates everything and exits, so the VRAM is freed.
    - design: a voice from a text description (scripts/design_voice.py, once)
    - clone:  every scene with the voice of a reference WAV (the designed one, or yours)

edge-tts (voice.mode=edge): Microsoft Edge's neural voices. No GPU, and it returns
the time of every word. But it needs internet (the text goes to Microsoft), it is
not an official API and a generic voice does not meet YouTube's "inauthentic
content" policy: for TESTS only.
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

from llmkit import config

TICKS_PER_SECOND = 10_000_000          # edge-tts times are in 100 ns units
QWEN_WORKER = config.ROOT / "src" / "tts_worker" / "qwen_worker.py"


# --------------------------------------------------------------- Qwen3-TTS
def run_qwen(job, python, timeout=3600):
    """Runs one job in the TTS environment (see qwen_worker.py for its format) -> its result."""
    python = Path(python) if Path(python).is_absolute() else config.BASE / python
    if not python.exists():
        raise RuntimeError(f"TTS environment not found: {python}. Create it as the README says "
                           "(uv venv .venv-tts --python 3.12 + torch cu128 + qwen-tts).")
    with tempfile.TemporaryDirectory() as folder:
        job_file = Path(folder) / "job.json"
        job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([str(python), str(QWEN_WORKER), str(job_file)], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Qwen3-TTS failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def clone_qwen(texts, outs, reference, reference_text, options):
    """Every text with the reference voice -> WAVs in `outs` (one job: the model loads once)."""
    return run_qwen({"task": "clone", "model": options["model"], "language": options.get("language", "Spanish"),
                     "ref_audio": str(reference), "ref_text": reference_text, "seed": options.get("seed"),
                     "batch_size": options.get("batch_size", 4),
                     "items": [{"text": t, "out": str(o)} for t, o in zip(texts, outs)]}, options["python"])


def design_qwen(text, descriptions, outs, options):
    """One candidate voice per description, all reading `text`."""
    return run_qwen({"task": "design", "model": options["design_model"],
                     "language": options.get("language", "Spanish"), "seed": options.get("seed"),
                     "batch_size": options.get("batch_size", 4),
                     "items": [{"text": text, "instruct": d, "out": str(o)} for d, o in zip(descriptions, outs)]},
                    options["python"])


# ---------------------------------------------------------------- edge-tts


def synthesize_edge(text, mp3_path, voice="es-ES-AlvaroNeural", rate="+0%", pitch="+0Hz", attempts=3):
    """Writes the MP3 and returns the words with their times: [{"word", "start", "end"}, ...] (seconds)."""
    import edge_tts

    for attempt in range(1, attempts + 1):
        try:
            words = []
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
            with open(mp3_path, "wb") as f:
                for chunk in communicate.stream_sync():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        start = chunk["offset"] / TICKS_PER_SECOND
                        words.append({"word": chunk["text"], "start": round(start, 3),
                                      "end": round(start + chunk["duration"] / TICKS_PER_SECOND, 3)})
            return words
        except Exception:          # network hiccups: the service sometimes drops the connection
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
