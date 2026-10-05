"""beats 🤖: WHAT is shown and FROM WHICH WORD -> state["beats"].

The narration is cut where what has to be shown changes (a person, a place, a figure, an
idea), not by sentences: a scene can have several beats. For each one the LLM says what
it needs (kind: entity | generic | data | source | abstract), what must be seen (subject),
how to search for it (query) and an image prompt in case nothing good is found
(prompts/beats.md). It sees the whole script and how long each scene lasts.

Code then finds the word each beat starts at (its first words, copied by the model),
takes that word's time from the align node, and merges beats shorter than
visuals.min_beat_seconds into the previous one.

Saved to <run>/beats.json. `--redo beats` reads it again if it exists (edit it: cuts,
subjects, queries, prompts) and then redoes the visuals; delete it for new beats.
"""
import json

from graph.nodes.align import normalize
from graph.state import BeatsDraft, Script
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "beats"
SCENES_PER_CALL = 8           # scenes per LLM call (the whole script always goes as context)
LEAD = 0.1                    # a beat's image appears this much before its first word


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


def scene_beats(scene, drafts, words, scene_seconds, min_seconds, style):
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
                      "query": d.query if d else "",
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
    ends = [b["start"] for b in beats[1:]] + [scene_seconds]
    for i, b in enumerate(beats):
        b["seconds"] = round(ends[i] - b["start"], 3)
        last = beats[i + 1]["word"] if i + 1 < len(beats) else len(narration)
        b["text"] = " ".join(narration[b["word"]:last])
    return beats


def make_beats(hub, settings=None):
    settings = settings or load_settings()
    write = hub.chain(load_prompt(NODE), schema=BeatsDraft, node=NODE)

    def beats(state, config):
        options = settings["visuals"]
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        beats_file = runs_dir(run_id) / "beats.json"
        if beats_file.exists():                                 # yours (or a previous run's): kept
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
        ids = [s.id for s in script.scenes]
        groups = [ids[i:i + SCENES_PER_CALL] for i in range(0, len(ids), SCENES_PER_CALL)]
        for attempt in range(2):                                # the scenes it skips: asked once more
            for group in groups:
                wanted = [i for i in group if i not in drafts]
                if wanted:
                    draft = write.invoke({"text": beats_request(state, script, audio, wanted, style, seconds)})
                    drafts |= {s.scene: s.beats for s in draft.scenes if s.scene in wanted}

        result = []
        for scene in script.scenes:
            result += scene_beats(scene, drafts.get(scene.id), audio[scene.id]["words"],
                                  audio[scene.id]["duration"] + gap, options.get("min_beat_seconds", 1.5), style)
        for n, beat in enumerate(result, start=1):
            beat["id"] = n
        result = [{"id": b["id"], **{k: v for k, v in b.items() if k != "id"}} for b in result]
        beats_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        kinds = [b["kind"] for b in result]
        print(f"  · beats: {len(result)} fragmentos en {len(script.scenes)} escenas · "
              + ", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds)))
        return {"beats": result}

    return beats
