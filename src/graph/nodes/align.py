"""align: the time of EVERY word of the narration -> state["audio"][scene]["words"].

Whisper (services/workers.py -> align_worker.py, on the GPU) transcribes each scene's WAV
with word times. Its words are then matched to the script's own words (difflib): a word
Whisper wrote differently ("1.144" for "mil ciento cuarenta y cuatro") gets its time
interpolated between its neighbours. So words[i] is always the script's i-th word: the
beats point at word indices, and the subtitles will use the same list.

Scenes that already have one time per word (voice.mode=edge) are not transcribed again.
"""
import difflib
import re
import unicodedata

from graph.state import Script
from services import workers
from services.settings import load_settings


def normalize(word):
    plain = unicodedata.normalize("NFKD", word.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain)


def match_words(script_words, heard, duration):
    """script_words: the narration split by spaces; heard: [{"word", "start", "end"}] from Whisper.
    -> [{"word", "start", "end"}] with one entry per script word."""
    a = [normalize(w) for w in script_words]
    b = [normalize(h["word"]) for h in heard]
    times = [None] * len(a)
    for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            h = heard[block.b + k]
            times[block.a + k] = (h["start"], h["end"])

    # Unmatched runs: share the gap between the known neighbours, by the length of each word
    i = 0
    while i < len(a):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(a) and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else (heard[0]["start"] if heard else 0.0)
        right = times[j][0] if j < len(a) else (heard[-1]["end"] if heard else duration)
        right = max(right, left)
        weights = [max(1, len(a[k])) for k in range(i, j)]
        t = left
        for k, w in zip(range(i, j), weights):
            step = (right - left) * w / sum(weights)
            times[k] = (round(t, 3), round(t + step, 3))
            t += step
        i = j
    return [{"word": w, "start": s, "end": e} for w, (s, e) in zip(script_words, times)]


def make_align(settings=None):
    settings = settings or load_settings()

    def align(state):
        options = settings.get("align", {})
        script = Script.model_validate(state["script"])
        audio = {k: dict(v) for k, v in state["audio"].items()}
        todo = [s for s in script.scenes if len(audio[s.id].get("words") or []) != len(s.narration.split())]
        if todo:
            result = workers.run("align", {"model": options.get("model", "openai/whisper-large-v3-turbo"),
                                           "language": options.get("language", "spanish"),
                                           "items": [{"audio": audio[s.id]["path"]} for s in todo]}, settings)
            for scene, item in zip(todo, result["items"]):
                audio[scene.id]["words"] = match_words(scene.narration.split(), item["words"], item["duration"])
            print(f"  · align: {len(todo)} escenas con Whisper ({result['seconds']} s)")
        return {"audio": audio}

    return align
