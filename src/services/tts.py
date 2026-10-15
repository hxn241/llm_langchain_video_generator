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

spoken(): the figures of a text in words, for the voice only (the script and the subtitles keep
"1917"; align matches what Whisper hears to the script's words and fills in the rest).
"""
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from num2words import num2words

from llmkit import config

TICKS_PER_SECOND = 10_000_000          # edge-tts times are in 100 ns units
QWEN_WORKER = config.ROOT / "src" / "tts_worker" / "qwen_worker.py"


# ------------------------------------------------------------ figures in words
# 57.600 / 1917 / 3,26 / 0,05, optionally followed by % ("1.000" with a dot = thousands, Spanish style)
FIGURE = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(\s?%)?(?![\w])")


# The noun after a figure decides its gender ("seiscientas mariposas", "veintiún años"): num2words
# only says the masculine. Feminine: -a/-as, -ción, -dad... except these (and a few -e/-es that are)
MASCULINE_A = {"día", "mapa", "problema", "sistema", "programa", "tema", "planeta", "clima", "idioma", "dilema",
               "esquema", "teorema", "poema", "diagrama", "cometa", "fantasma", "genoma", "aroma", "pijama"}
FEMININE_OTHER = {"vez", "especie", "imagen", "noche", "red", "ley", "mujer", "frase", "clase", "fase", "base",
                  "llave", "nave", "ave", "fuente", "muerte", "parte", "piel", "sal", "luz", "voz", "nariz",
                  "raíz", "cruz", "calle", "torre", "carne", "nube", "gente", "serpiente", "suerte", "tarde"}


NOT_NOUNS = {"de", "del", "a", "al", "la", "las", "el", "los", "lo", "un", "una", "unos", "unas", "y", "e", "o",
             "u", "en", "con", "por", "para", "que", "se", "su", "sus", "ya", "fue", "era", "es", "no", "ni",
             "más", "menos", "hasta", "desde", "sin", "como", "cuando", "esa", "ese", "esta", "este"}


def _singular(word):
    return re.sub(r"(ces)$", "z", word) if word.endswith("ces") else re.sub(r"(?<=[^aeiou])es$|s$", "", word)


def _is_feminine(noun):
    noun = noun.lower()
    base = _singular(noun)
    if noun in MASCULINE_A or base in MASCULINE_A:
        return False
    return (noun in FEMININE_OTHER or base in FEMININE_OTHER or noun.endswith(("a", "as"))
            or re.search(r"(ción|sión|ciones|siones|dad|dades|tud|tudes)$", noun) is not None)


def _feminine(words):
    """'cincuenta y siete mil seiscientos' -> '... seiscientas'; 'veintiuno' -> 'veintiuna'. What comes
    before millón/millones stays masculine ("doscientos millones de personas")."""
    head, sep, tail = words.rpartition(" millones ") if " millones " in words else words.rpartition(" millón ")
    tail = re.sub(r"cientos\b", "cientas", tail)
    tail = re.sub(r"(?<!\w)(veinti)?uno\b", lambda m: (m.group(1) or "") + "una", tail)
    return head + sep + tail


def _number(digits):
    return num2words(int(digits), lang="es")


def _figure(match, text):
    whole, decimals, percent = match.group(1).replace(".", ""), match.group(2), match.group(3)
    words = _number(whole)
    noun = re.match(r"\s+([a-záéíóúñü]+)", text[match.end():], re.IGNORECASE)
    if decimals:                         # 3,26 -> "tres coma veintiséis"; 0,05 -> "cero coma cero cinco"
        zeros = len(decimals) - len(decimals.lstrip("0"))
        words += " coma " + " ".join(["cero"] * zeros + ([_number(decimals)] if decimals.strip("0") else []))
    elif not percent and noun and noun.group(1).lower() not in NOT_NOUNS:   # "en 2021 las aves": a year
        if _is_feminine(noun.group(1)):
            words = _feminine(words)                                 # "57.600 mariposas", "21 especies"
        else:
            words = re.sub(r"veintiuno$", "veintiún", re.sub(r"(?<!\w)uno$", "un", words))   # "21 años"
    return words + (" por ciento" if percent else "")


def spoken(text):
    """The figures of `text` as they are said in Spain: "en 1917" -> "en mil novecientos diecisiete",
    "1,4 microgramos" -> "uno coma cuatro microgramos", "25 %" -> "veinticinco por ciento"."""
    return FIGURE.sub(lambda m: _figure(m, text), text)


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
                     "batch_size": options.get("batch_size", 4), "generation": options.get("generation") or {},
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
