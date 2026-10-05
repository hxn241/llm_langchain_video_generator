"""Whisper worker: the time of every word heard in some WAVs. Runs with .venv-tts (torch +
transformers), NOT with the project's venv:

    .venv-tts/Scripts/python.exe src/tts_worker/align_worker.py job.json

job.json:
    {"model": "openai/whisper-large-v3-turbo", "language": "spanish",
     "items": [{"audio": "scene_01.wav"}, ...]}

Prints one JSON line at the end:
    {"ok": true, "items": [{"audio": "...", "duration": 6.3,
                            "words": [{"word": "Hace", "start": 0.0, "end": 0.32}, ...]}], "seconds": ...}
It loads the model once, transcribes every item and exits (the VRAM is freed).
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as F  # noqa: E402
from transformers import pipeline  # noqa: E402

RATE = 16_000


def load(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    if rate != RATE:
        audio = F.resample(audio, rate, RATE)
    return audio.numpy(), len(audio) / RATE


def main(job_path):
    job = json.loads(open(job_path, encoding="utf-8-sig").read())
    started = time.time()
    asr = pipeline("automatic-speech-recognition", model=job["model"], torch_dtype=torch.float16, device="cuda:0")
    items = []
    for item in job["items"]:
        audio, duration = load(item["audio"])
        out = asr({"raw": audio, "sampling_rate": RATE}, return_timestamps="word",
                  generate_kwargs={"language": job.get("language", "spanish"), "task": "transcribe"})
        words = []
        for chunk in out.get("chunks", []):
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()
            if text:
                start = 0.0 if start is None else float(start)
                words.append({"word": text, "start": round(start, 3),
                              "end": round(float(end) if end is not None else duration, 3)})
        items.append({"audio": item["audio"], "duration": round(duration, 3), "words": words})
    print(json.dumps({"ok": True, "items": items, "seconds": round(time.time() - started, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
