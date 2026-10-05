"""SigLIP 2 worker: how well each image matches a text (cosine similarity of their embeddings).
Runs with .venv-tts (torch + transformers), NOT with the project's venv:

    .venv-tts/Scripts/python.exe src/tts_worker/embed_worker.py job.json

job.json:
    {"model": "google/siglip2-base-patch16-224",
     "items": [{"text": "young couple checking bills in a kitchen", "images": ["a.jpg", "b.jpg"]}, ...]}

Prints one JSON line at the end: {"ok": true, "scores": [[0.12, 0.08], ...], "seconds": ...}
(one list per item, one score per image, in order; higher = closer). A broken image scores -1.
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

BATCH = 16


def features(output):
    """get_*_features returns a tensor (or, in some versions, an output with pooler_output)."""
    return output if torch.is_tensor(output) else output.pooler_output


def main(job_path):
    job = json.loads(open(job_path, encoding="utf-8-sig").read())
    started = time.time()
    model = AutoModel.from_pretrained(job["model"], torch_dtype=torch.float16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(job["model"])
    scores = []
    with torch.no_grad():
        for item in job["items"]:
            text = processor(text=[item["text"]], padding="max_length", max_length=64, truncation=True,
                             return_tensors="pt").to("cuda")
            t = torch.nn.functional.normalize(features(model.get_text_features(**text)).float(), dim=-1)
            row = []
            for i in range(0, len(item["images"]), BATCH):
                paths = item["images"][i:i + BATCH]
                images, ok = [], []
                for p in paths:
                    try:
                        images.append(Image.open(p).convert("RGB"))
                        ok.append(True)
                    except Exception:
                        ok.append(False)
                sims = []
                if images:
                    pixels = processor(images=images, return_tensors="pt").to("cuda")
                    pixels = {k: (v.half() if v.is_floating_point() else v) for k, v in pixels.items()}
                    v = torch.nn.functional.normalize(features(model.get_image_features(**pixels)).float(), dim=-1)
                    sims = (v @ t.T).squeeze(-1).tolist()
                    sims = sims if isinstance(sims, list) else [sims]
                it = iter(sims)
                row += [round(next(it), 4) if good else -1.0 for good in ok]
            scores.append(row)
    print(json.dumps({"ok": True, "scores": scores, "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main(sys.argv[1])
