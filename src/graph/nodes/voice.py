"""voice: one normalized WAV per scene -> state["audio"][scene_id] = {path, duration, words}.

voice.mode in config.yaml:
- clone:  Qwen3-TTS with the voice of voice.reference (a designed voice, or yours). Local, GPU.
          All scenes in ONE job: the model loads once and the VRAM is freed when it ends.
- edge:   edge-tts predefined voice (tests; see services/tts.py). Gives word timings.
- record: you record each scene (README; not implemented yet)

The voice gets the figures in words (tts.spoken: "1917" -> "mil novecientos diecisiete"; read as
digits, the 2026-10-15 butterflies voice said the year oddly); the script keeps them.

The silence the TTS leaves at the end is cut, so the next scene's image changes
right when the idea changes (the pause between scenes is added in assemble).
Without word timings (clone), `words` stays empty: the subtitles node will get
them with faster-whisper.
"""
from pathlib import Path

from graph.state import Script
from llmkit.config import BASE
from services import media, tts
from services.settings import load_settings, runs_dir

TAIL = 0.2          # seconds kept after the last word (so it does not sound clipped)


def _project_path(path):
    """A path from config.yaml: relative ones are taken from the project's folder."""
    return Path(path) if Path(path).is_absolute() else BASE / path


def make_voice(settings=None):
    settings = settings or load_settings()

    def voice(state, config):
        options, lufs = settings["voice"], settings["video"]["loudness_lufs"]
        script = Script.model_validate(state["script"])
        run_id = state.get("run_id") or config["configurable"]["thread_id"]     # a run = a thread
        folder = runs_dir(run_id) / "audio"
        folder.mkdir(exist_ok=True)
        wav = {s.id: folder / f"scene_{s.id:02d}.wav" for s in script.scenes}
        words = {s.id: [] for s in script.scenes}

        if options["mode"] == "clone":
            reference = _project_path(options["reference"])
            reference_text = _project_path(options["reference_text"])
            if not reference.exists() or not reference_text.exists():
                raise FileNotFoundError(f"Missing the reference voice ({options['reference']} and "
                                        f"{options['reference_text']}): run scripts/design_voice.py")
            raw = {s.id: folder / f"scene_{s.id:02d}.raw.wav" for s in script.scenes}
            tts.clone_qwen([tts.spoken(s.narration) for s in script.scenes], [raw[s.id] for s in script.scenes],
                           reference, reference_text.read_text(encoding="utf-8").strip(), options["clone"])
            for s in script.scenes:
                media.normalize_loudness(raw[s.id], wav[s.id], lufs, trim_silence=True)
                raw[s.id].unlink()

        elif options["mode"] == "edge":
            for s in script.scenes:
                raw = folder / f"scene_{s.id:02d}.mp3"
                words[s.id] = tts.synthesize_edge(tts.spoken(s.narration), raw, **options["edge"])
                end = words[s.id][-1]["end"] + TAIL if words[s.id] else None
                media.normalize_loudness(raw, wav[s.id], lufs, end=end)
                raw.unlink()

        else:
            raise NotImplementedError(f"voice.mode={options['mode']!r} is not implemented yet "
                                      "(available: clone, edge)")

        return {"audio": {s.id: {"path": str(wav[s.id]), "duration": round(media.wav_duration(wav[s.id]), 3),
                                 "words": words[s.id]} for s in script.scenes}}

    return voice
