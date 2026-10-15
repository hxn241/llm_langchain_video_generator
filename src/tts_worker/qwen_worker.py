"""Qwen3-TTS worker. Runs with .venv-tts (torch + qwen-tts), NOT with the project's venv:

    .venv-tts/Scripts/python.exe src/tts_worker/qwen_worker.py job.json

It loads the model, generates every item of the job, writes the WAVs and exits
(so the VRAM is freed when it ends). It imports nothing from the project.

job.json:
    {"task": "design" | "clone",
     "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",     # VoiceDesign model for "design"
     "language": "Spanish",
     "items": [{"text": "...", "out": "path.wav", "instruct": "..."}],   # instruct: only "design"
     "ref_audio": "voice_reference.wav", "ref_text": "...",              # only "clone"
     "seed": 1234, "batch_size": 4,
     "generation": {"temperature": 0.7, "top_k": 30, ...}}               # only "clone": sampling options
                                                                         # (model default: 0.9 / 50)

Prints one JSON line at the end: {"ok": true, "files": [...], "seconds": ...}
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")      # Windows without developer mode

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main(job_path):
    job = json.loads(open(job_path, encoding="utf-8-sig").read())     # -sig: also files saved with a BOM
    started = time.time()
    if job.get("seed") is not None:
        torch.manual_seed(job["seed"])

    model = Qwen3TTSModel.from_pretrained(job["model"], device_map="cuda:0", dtype=torch.bfloat16,
                                          attn_implementation="sdpa")    # flash-attn: hard on Windows
    language = job.get("language", "Spanish")
    prompt = None
    if job["task"] == "clone":       # the reference voice is processed ONCE for every item
        prompt = model.create_voice_clone_prompt(ref_audio=job["ref_audio"], ref_text=job["ref_text"])

    files = []
    for batch in batches(job["items"], job.get("batch_size", 4)):
        texts, languages = [i["text"] for i in batch], [language] * len(batch)
        if job["task"] == "design":
            wavs, sr = model.generate_voice_design(text=texts, language=languages,
                                                   instruct=[i["instruct"] for i in batch])
        elif job["task"] == "clone":
            wavs, sr = model.generate_voice_clone(text=texts, language=languages, voice_clone_prompt=prompt,
                                                  **job.get("generation", {}))
        else:
            raise ValueError(f"Unknown task: {job['task']!r}")
        for item, wav in zip(batch, wavs):
            sf.write(item["out"], wav, sr)
            files.append(item["out"])

    print(json.dumps({"ok": True, "files": files, "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main(sys.argv[1])
