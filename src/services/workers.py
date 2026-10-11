"""GPU workers: scripts in src/tts_worker/ that run with .venv-tts (torch + transformers), so
the project's venv needs no torch. Each call loads its model, does the whole job and exits
(the VRAM is freed when it ends).

    align  (align_worker.py)  Whisper: the time of every word of some WAVs
    embed  (embed_worker.py)  SigLIP 2: how well each image matches a text
    parallax (parallax_worker.py)  Depth Anything: photos -> 2.5D video segments

    result = run("align", {"model": ..., "items": [...]})

The Python of the environment: `workers.python` in config.yaml (default: voice.clone.python).
"""
import json
import subprocess
import tempfile
from pathlib import Path

from llmkit import config
from services.settings import load_settings

FOLDER = config.ROOT / "src" / "tts_worker"


def python(settings=None):
    settings = settings or load_settings()
    path = (settings.get("workers") or {}).get("python") or settings["voice"]["clone"]["python"]
    path = Path(path) if Path(path).is_absolute() else config.BASE / path
    if not path.exists():
        raise RuntimeError(f"GPU environment not found: {path}. Create it as the README says "
                           "(uv venv .venv-tts --python 3.12 + torch cu128 + qwen-tts).")
    return path


def run(worker, job, settings=None, timeout=3600):
    """Runs src/tts_worker/<worker>_worker.py with `job` -> its result (the last JSON line it prints)."""
    with tempfile.TemporaryDirectory() as folder:
        job_file = Path(folder) / "job.json"
        job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([str(python(settings)), str(FOLDER / f"{worker}_worker.py"), str(job_file)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Worker {worker!r} failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])
