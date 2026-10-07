"""beats 🤖: WHAT is shown and FROM WHICH WORD -> state["beats"].

The narration is cut where what has to be shown changes (a person, a place, a figure, an
idea), not by sentences: a scene can have several beats. For each one the LLM says what
it needs (kind: entity | generic | data | source | abstract), what must be seen (subject),
how to search for it (queries: 2-3, most concrete first) and an image prompt in case nothing good is found
(prompts/beats.md). It sees the whole script and how long each scene lasts.

Code then finds the word each beat starts at (its first words, copied by the model),
takes that word's time from the align node, merges beats shorter than
visuals.min_beat_seconds into the previous one and splits those longer than the format's
maximum (visuals.beat_seconds) into shots of about the same length: the model writes
longer beats than it is asked for, and a still shot of 6 s is too long.

Manual mode (graph/manual.py): a scene with your [file] marks becomes one beat per mark
(kind "manual", with its "path"), without the LLM; only the scenes without marks go to it.

Saved to <run>/beats.json. `--redo beats` reads it again if it exists (not in manual mode:
there your marks in guion.md are the source) (edit it: cuts,
subjects, queries, prompts) and then redoes the visuals; delete it for new beats.
"""
import json
import math
from pathlib import Path

from graph.nodes.align import normalize
from graph.state import BeatsDraft, Script
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "beats"
SCENES_PER_CALL = 8           # scenes per LLM call (the whole script always goes as context)
LEAD = 0.1                    # a beat's image appears this much before its first word
SPLIT_TOLERANCE = 1.2         # a beat up to 20 % over the maximum length is kept whole


def beats_request(state, script, audio, numbers, style, seconds):
    parts = [f"Tema del vídeo: {state.get('topic') or script.title}", f"Título: {script.title}"]
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    parts += [f"Duración de cada fragmento: entre {seconds[0]} y {seconds[1]} segundos.",
              f"Estilo (termina cada prompt con esto): {style}", "",
              "# Guion completo (número. [segundos] narración)"]
    parts += [f"{s.id}. [{audio[s.id]['duration']:.0f} s] {s.narration}" for s in script.scenes]
    parts += ["", f"# Corta en fragmentos las escenas: {', '.join(map(str, numbers))}"]
    return "\n".join(parts)


def locate(phrase, words, after):
    """Index (> after) where `phrase` starts among `words` (normalized), trying its first 3, 2, 1
    words; None if it is not there."""
    wanted = [w for w in (normalize(p) for p in phrase.split()) if w]
    plain = [normalize(w) for w in words]
    for k in (3, 2, 1):
        if len(wanted) < k:
            continue
        for i in range(after + 1, len(plain) - k + 1):
            if plain[i:i + k] == wanted[:k]:
                return i
    return None


def split_long(beats, words, word_count, scene_seconds, max_seconds):
    """A beat longer than max_seconds (by more than SPLIT_TOLERANCE) becomes ceil(length / max)
    shots of about the same length, cut at the word closest to each point. Each part keeps the
    beat's description: visuals gives it another clip (a clip is used once per video)."""
    ends = [b["start"] for b in beats[1:]] + [scene_seconds]
    result = []
    for i, (b, end) in enumerate(zip(beats, ends)):
        seconds = end - b["start"]
        parts = math.ceil(seconds / max_seconds) if seconds > max_seconds * SPLIT_TOLERANCE else 1
        last_word = beats[i + 1]["word"] if i + 1 < len(beats) else min(word_count, len(words))
        cuts = [b["word"]]
        for k in range(1, parts):
            target = b["start"] + k * seconds / parts
            choices = range(cuts[-1] + 2, last_word - 1)           # at least 2 words on each side
            if not choices:
                break
            cuts.append(min(choices, key=lambda i: abs(words[i]["start"] - LEAD - target)))
        result += [dict(b, word=c, start=b["start"] if n == 0 else round(max(0.0, words[c]["start"] - LEAD), 3))
                   for n, c in enumerate(cuts)]
    return result


def scene_beats(scene, drafts, words, scene_seconds, min_seconds, max_seconds, style):
    """The drafts of one scene -> [beat] with word index, start and length (ids are set later)."""
    narration = scene.narration.split()
    if not drafts:
        drafts = []
    placed = []
    for n, d in enumerate(drafts):
        index = 0 if n == 0 else locate(d.start, narration, placed[-1][0] if placed else 0)
        if index is None:
            continue                                   # not found: its words stay with the previous beat
        placed.append((index, d))
    if not placed:                                     # nothing usable: the whole scene, one beat
        placed = [(0, None)]

    beats = []
    for index, d in placed:
        start = 0.0 if index == 0 else max(0.0, words[index]["start"] - LEAD)
        beats.append({"scene": scene.id, "word": index, "start": round(start, 3),
                      "kind": d.kind if d else "generic",
                      "subject": d.subject if d else scene.narration,
                      "queries": d.queries if d else [],
                      "prompt": d.prompt if d else f"Documentary photo illustrating: {scene.narration}. {style}"})
    # Too short to be seen: it takes the words of the NEXT beat (it is where something new
    # starts, so its image is the one that matters); the last one joins the one before.
    while len(beats) > 1:
        ends = [b["start"] for b in beats[1:]] + [scene_seconds]
        short = [i for i, b in enumerate(beats) if ends[i] - b["start"] < min_seconds]
        if not short:
            break
        i = short[0]
        del beats[i + 1 if i + 1 < len(beats) else i]
    beats = split_long(beats, words, len(narration), scene_seconds, max_seconds)
    return finish(beats, narration, scene_seconds)


def finish(beats, narration, scene_seconds):
    """Each beat's length (to the next one or the scene's end) and the words it covers."""
    ends = [b["start"] for b in beats[1:]] + [scene_seconds]
    for i, b in enumerate(beats):
        b["seconds"] = round(ends[i] - b["start"], 3)
        last = beats[i + 1]["word"] if i + 1 < len(beats) else len(narration)
        b["text"] = " ".join(narration[b["word"]:last])
    return beats


def manual_beats(scene, marks, words, scene_seconds):
    """Your marks of one scene (manual mode) -> [beat] of kind "manual", with the file to show.
    Kept as you put them: not merged, not split."""
    narration = scene.narration.split()
    beats = [{"scene": scene.id, "word": m["word"],
              "start": 0.0 if m["word"] == 0 else
              round(max(0.0, words[min(m["word"], len(words) - 1)]["start"] - LEAD), 3),
              "kind": "manual", "subject": Path(m["path"]).name, "queries": [], "prompt": "", "path": m["path"]}
             for m in sorted(marks, key=lambda m: m["word"])]
    beats = [b for i, b in enumerate(beats) if i + 1 == len(beats) or beats[i + 1]["word"] != b["word"]]
    return finish(beats, narration, scene_seconds)


def make_beats(hub, settings=None):
    settings = settings or load_settings()
    write = hub.chain(load_prompt(NODE), schema=BeatsDraft, node=NODE)

    def beats(state, config):
        options = settings["visuals"]
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        beats_file = runs_dir(run_id) / "beats.json"
        # Manual mode: always from your marks (guion.md is the source); otherwise yours/a previous run's
        marks = {int(k): v for k, v in (state.get("manual_visuals") or {}).items()}
        if beats_file.exists() and not state.get("manual"):
            found = json.loads(beats_file.read_text(encoding="utf-8"))
            print(f"  · beats: {len(found)} fragmentos de {beats_file.name} (bórralo para hacerlos de nuevo)")
            return {"beats": found}

        video_format = state.get("format") or settings.get("format", "long")
        script = Script.model_validate(state["script"])
        audio = state["audio"]
        style = options.get("image_style", "")
        seconds = options.get("beat_seconds", {}).get(video_format, [2, 6])
        gap = settings["video"].get("scene_gap", 0.15)

        drafts = {}
        ids = [s.id for s in script.scenes if s.id not in marks]          # scenes with your marks: no LLM
        groups = [ids[i:i + SCENES_PER_CALL] for i in range(0, len(ids), SCENES_PER_CALL)]
        for attempt in range(2):                                # the scenes it skips: asked once more
            for group in groups:
                wanted = [i for i in group if i not in drafts]
                if wanted:
                    draft = write.invoke({"text": beats_request(state, script, audio, wanted, style, seconds)})
                    drafts |= {s.scene: s.beats for s in draft.scenes if s.scene in wanted}

        result = []
        for scene in script.scenes:
            words, scene_seconds = audio[scene.id]["words"], audio[scene.id]["duration"] + gap
            if scene.id in marks:
                result += manual_beats(scene, marks[scene.id], words, scene_seconds)
            else:
                result += scene_beats(scene, drafts.get(scene.id), words, scene_seconds,
                                      options.get("min_beat_seconds", 1.5), seconds[1], style)
        for n, beat in enumerate(result, start=1):
            beat["id"] = n
        result = [{"id": b["id"], **{k: v for k, v in b.items() if k != "id"}} for b in result]
        beats_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        kinds = [b["kind"] for b in result]
        print(f"  · beats: {len(result)} fragmentos en {len(script.scenes)} escenas · "
              + ", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds)))
        return {"beats": result}

    return beats
