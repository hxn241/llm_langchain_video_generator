"""Designs the channel's voice with Qwen3-TTS VoiceDesign (once).

    python scripts/design_voice.py                       # one candidate per description below
    python scripts/design_voice.py --describe "Male, 50 years old, ..." --describe "..."
    python scripts/design_voice.py --pick 3              # candidate 3 -> assets/voice_reference.wav

The chosen candidate becomes the reference that voice.mode=clone copies in every
scene, so the voice is the same in every scene and every video. To use YOUR voice
instead, record ~10-15 s into assets/voice_reference.wav and write exactly what you
said in assets/voice_reference.txt.
"""
import argparse
import json
import shutil

from llmkit import config
from services import media, tts
from services.settings import load_settings

FOLDER = config.DATA_DIR / "samples" / "voice_design"

# What every candidate reads: ~13 s, varied sounds, the channel's tone. It is also the
# transcript of the reference, so it must match the audio EXACTLY.
SAMPLE_TEXT = ("Hola, bienvenidos. Hoy os cuento una noticia de tecnología que me ha sorprendido mucho, "
               "porque cambia la forma en la que usamos la inteligencia artificial. "
               "Vamos a ver qué ha pasado y por qué importa.")

DESCRIPTIONS = [
    "Male, about 30 years old, native Spanish speaker from Spain with a Castilian accent. Warm mid-low "
    "voice, clear diction, friendly and engaged like a tech YouTuber explaining news, natural pace.",
    "Male, about 40 years old, native Spanish speaker from Spain with a Castilian accent. Deep, calm, "
    "authoritative voice like a documentary narrator, measured pace.",
    "Male, about 25 years old, native Spanish speaker from Spain with a Castilian accent. Bright, energetic "
    "voice, lively and enthusiastic delivery, slightly fast pace.",
    "Female, about 30 years old, native Spanish speaker from Spain with a Castilian accent. Warm, clear "
    "voice, friendly and confident like a science communicator, natural pace.",
    "Female, about 35 years old, native Spanish speaker from Spain with a Castilian accent. Calm, "
    "professional voice like a news anchor, crisp diction.",
    "Female, about 25 years old, native Spanish speaker from Spain with a Castilian accent. Bright, upbeat, "
    "expressive voice, lively pace.",
]


def generate(descriptions, settings):
    """New candidates are numbered after the existing ones (earlier ones are kept)."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    index = FOLDER / "candidates.json"
    candidates = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    first = max(map(int, candidates), default=0) + 1
    numbers = range(first, first + len(descriptions))
    raws = [FOLDER / f"{n:02d}.raw.wav" for n in numbers]
    print(f"Generating {len(descriptions)} voices (the first time it downloads the model, ~4 GB)...")
    result = tts.design_qwen(SAMPLE_TEXT, descriptions, raws, settings["voice"]["clone"])
    for n, raw, description in zip(numbers, raws, descriptions):
        wav = FOLDER / f"{n:02d}.wav"
        media.normalize_loudness(raw, wav, settings["video"]["loudness_lufs"], trim_silence=True)
        raw.unlink()
        candidates[str(n)] = description
        print(f"  {n:02d}.wav  {media.wav_duration(wav):4.1f} s  {description[:90]}...")
    index.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{result['seconds']} s. Listen to them in {config.readable_path(FOLDER)} and choose one with --pick N")


def pick(number, settings):
    candidate = FOLDER / f"{number:02d}.wav"
    if not candidate.exists():
        raise SystemExit(f"{config.readable_path(candidate)} does not exist: generate the candidates first")
    reference = config.BASE / settings["voice"]["reference"]
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidate, reference)
    (config.BASE / settings["voice"]["reference_text"]).write_text(SAMPLE_TEXT, encoding="utf-8")
    print(f"Voice {number:02d} -> {settings['voice']['reference']} (voice.mode=clone will use it)")


def main():
    parser = argparse.ArgumentParser(description="Diseña la voz del canal con Qwen3-TTS VoiceDesign.")
    parser.add_argument("--describe", action="append", help="voice description (English works best); repeatable")
    parser.add_argument("--variants", type=int, default=1, metavar="K",
                        help="K candidates per description (each generation sounds a bit different)")
    parser.add_argument("--pick", type=int, metavar="N", help="use candidate N as the reference voice")
    args = parser.parse_args()
    settings = load_settings()
    if args.pick:
        pick(args.pick, settings)
    else:
        generate([d for d in (args.describe or DESCRIPTIONS) for _ in range(args.variants)], settings)


if __name__ == "__main__":
    from services import run_cli
    run_cli(main)
