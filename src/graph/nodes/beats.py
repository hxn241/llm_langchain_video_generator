"""beats 🤖: WHAT is shown and FROM WHICH WORD -> state["beats"].

The LLM plans by IDEA, not by sentence fragments: each scene gets 1-2 ideas (what that stretch
tells), inside the video's brief (nodes/brief.py: visual universe, what never appears, how each
metaphor is shown). For each idea it says what it needs (kind: entity | generic | data | source |
abstract), what must not be seen (must_not_show) and its different shots, each with what must be
seen in it (must_show), how to search for it (queries: 1-3, most concrete first) and an image
prompt in case nothing good is found (prompts/beats.md). Searches and must_show belong to the
SHOT, not the idea: the three parts of an idea shown as a full stand, a referee and a locker
room search one each (with the idea's, all three showed a stand: the first search ranks first).
It sees the whole script and how long each scene lasts.

Code then finds the word each idea starts at (its first words, copied by the model),
takes that word's time from the align node, merges beats shorter than
visuals.min_beat_seconds into the previous one and splits those longer than the format's
maximum (visuals.beat_seconds) into parts of about the same length: a still shot of 6 s is too
long. Each part of a split idea shows the next of its shots, never the same one twice: the
model is told how many shots each scene needs, and an idea is never cut into more parts than it
has distinct shots.

Manual mode (graph/manual.py): a scene with your [file] marks becomes one beat per mark
(kind "manual", with its "path"), without the LLM; only the scenes without marks go to it.

Saved to <run>/beats.json, plus <run>/manual/prompts_imagenes.md and plan_visual.json (every
beat's prompt, to make an image with your own AI and put it in montaje/ at the review pause).
`--redo beats` reads it again if it exists and is newer than
brief.json (not in manual mode: there your marks in guion.md are the source) (edit it: cuts,
subjects, queries, prompts) and then redoes the visuals; delete it for new beats.
"""
import json
import math
from pathlib import Path

from graph import manual
from graph.nodes.align import normalize
from graph.nodes.brief import brief_lines
from graph.state import BeatsDraft, Script
from prompts import load_prompt
from services.settings import load_settings, runs_dir

NODE = "beats"
SCENES_PER_CALL = 8           # scenes per LLM call (the whole script always goes as context)
LEAD = 0.1                    # a beat's image appears this much before its first word
SPLIT_TOLERANCE = 1.2         # a beat up to 20 % over the maximum length is kept whole
MAX_QUERY_WORDS = 3           # stock banks find nothing for longer searches ("light beam through copper cables")
QUERY_STOPWORDS = {"a", "an", "the", "of", "in", "on", "with", "through", "and", "to", "for", "at", "into",
                   "from", "by", "over", "under"}
NEGATIONS = {"no", "not", "without", "none", "never", "empty"}   # a search can't ask for an absence: dropped
                              # (cut to 3 words, "space radiator no water" became "space radiator no")
FIELDS = ("id", "scene", "word", "start", "seconds", "text", "kind", "idea", "must_show", "must_not_show",
          "subject", "queries", "prompt", "path")   # beats.json, in this order


def beats_request(state, script, audio, numbers, style, seconds):
    parts = [f"Tema del vídeo: {state.get('topic') or script.title}", f"Título: {script.title}"]
    if state.get("notes"):
        parts.append(f"Notas del autor: {state['notes']}")
    if state.get("brief"):
        parts += ["", "# Brief visual (todo plano sale de aquí)"] + brief_lines(state["brief"])
    parts += ["", f"Duración de cada plano: entre {seconds[0]} y {seconds[1]} segundos (una idea más larga "
                  "se ve con varios de sus planos seguidos).",
              f"Estilo (termina cada prompt con esto): {style}", "",
              "# Guion completo (número. [segundos · planos que necesita la escena] narración)",
              "Entre todas las ideas de una escena tiene que haber, como mínimo, tantos planos DISTINTOS "
              "como dice la escena: si una escena necesita 4 planos y tiene una sola idea, esa idea lleva "
              "4 planos; si tiene dos ideas, por ejemplo 2 y 2."]
    parts += [f"{s.id}. [{audio[s.id]['duration']:.0f} s · {shots_needed(audio[s.id]['duration'], seconds[1])} "
              f"planos] {s.narration}" for s in script.scenes]
    parts += ["", f"# Planifica las ideas de las escenas: {', '.join(map(str, numbers))}"]
    return "\n".join(parts)


def shots_needed(seconds, max_seconds):
    """How many shots a stretch of `seconds` is cut into (as split_long cuts it)."""
    return math.ceil(seconds / max_seconds) if seconds > max_seconds * SPLIT_TOLERANCE else 1


def short_queries(queries):
    """Searches of at most MAX_QUERY_WORDS words (without stopwords), without repeats, and without
    those that ask for an absence ("no", "without"...)."""
    split = (q.lower().split() for q in queries)
    short = (" ".join([w for w in words if w not in QUERY_STOPWORDS][:MAX_QUERY_WORDS])
             for words in split if not NEGATIONS & set(words))
    return list(dict.fromkeys(q for q in short if q))


def with_shot(beat):
    """A beat (or a part of a split idea) -> its own shot: part n shows shot n, with ITS searches
    and what must be seen in it (split_long never makes more parts than the idea has shots, so no
    shot is seen twice)."""
    shots = beat.pop("shots")
    shot = shots[min(beat.pop("part", 0), len(shots) - 1)]
    return beat | {"subject": shot["subject"], "prompt": shot["prompt"], "must_show": shot["must_show"],
                   "queries": short_queries(shot["queries"])}


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
    parts of about the same length, cut at the word closest to each point. Each part keeps the
    idea and is numbered (part) to show another of the idea's shots, with that shot's searches
    (and a clip is used once per video). Never more parts than the idea has
    distinct shots: a longer shot is better than the same image twice."""
    ends = [b["start"] for b in beats[1:]] + [scene_seconds]
    result = []
    for i, (b, end) in enumerate(zip(beats, ends)):
        seconds = end - b["start"]
        parts = min(shots_needed(seconds, max_seconds), len(b["shots"]))
        last_word = beats[i + 1]["word"] if i + 1 < len(beats) else min(word_count, len(words))
        cuts = [b["word"]]
        for k in range(1, parts):
            target = b["start"] + k * seconds / parts
            choices = range(cuts[-1] + 2, last_word - 1)           # at least 2 words on each side
            if not choices:
                break
            cuts.append(min(choices, key=lambda i: abs(words[i]["start"] - LEAD - target)))
        result += [dict(b, word=c, part=n,
                        start=b["start"] if n == 0 else round(max(0.0, words[c]["start"] - LEAD), 3))
                   for n, c in enumerate(cuts)]
    return result


def scene_beats(scene, ideas, words, scene_seconds, min_seconds, max_seconds, style):
    """The ideas of one scene -> [beat] with word index, start and length (ids are set later)."""
    narration = scene.narration.split()
    placed = []
    for n, d in enumerate(ideas or []):
        index = 0 if n == 0 else locate(d.start, narration, placed[-1][0] if placed else 0)
        if index is None:
            continue                                   # not found: its words stay with the previous idea
        placed.append((index, d))
    if not placed:                                     # nothing usable: the whole scene, one beat
        placed = [(0, None)]

    beats = []
    for index, d in placed:
        start = 0.0 if index == 0 else max(0.0, words[index]["start"] - LEAD)
        beats.append({"scene": scene.id, "word": index, "start": round(start, 3),
                      "kind": d.kind if d else "generic",
                      "idea": d.idea if d else scene.narration,
                      "must_not_show": d.must_not_show if d else [],
                      "shots": list({s.prompt.strip().lower(): s.model_dump() for s in d.shots}.values()) if d else
                               [{"subject": scene.narration, "must_show": [], "queries": [],
                                 "prompt": f"Documentary photo illustrating: {scene.narration}. {style}"}]})
    # Too short to be seen: it takes the words of the NEXT beat (it is where something new
    # starts, so its image is the one that matters); the last one joins the one before.
    while len(beats) > 1:
        ends = [b["start"] for b in beats[1:]] + [scene_seconds]
        short = [i for i, b in enumerate(beats) if ends[i] - b["start"] < min_seconds]
        if not short:
            break
        i = short[0]
        del beats[i + 1 if i + 1 < len(beats) else i]
    beats = [with_shot(b) for b in split_long(beats, words, len(narration), scene_seconds, max_seconds)]
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


def plain(beat):
    """The beat as beats made it. Runs from the old beat_images pause (your image per beat, before
    the search) saved it as kind "manual" with "planned_kind" and "path": that is undone, since
    your changes now go into <run>/montaje/ after visuals."""
    if not beat.get("planned_kind"):
        return beat
    return {k: v for k, v in beat.items() if k not in ("planned_kind", "path")} | {"kind": beat["planned_kind"]}


def make_beats(hub, settings=None):
    settings = settings or load_settings()
    write = hub.chain(load_prompt(NODE), schema=BeatsDraft, node=NODE)

    def beats(state, config):
        options = settings["visuals"]
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        beats_file, brief_file = runs_dir(run_id) / "beats.json", runs_dir(run_id) / "brief.json"
        # Manual mode: always from your marks (guion.md is the source); otherwise yours/a previous run's,
        # unless the brief is newer (a new or edited brief.json needs new beats)
        marks = {int(k): v for k, v in (state.get("manual_visuals") or {}).items()}
        stale = brief_file.exists() and beats_file.exists() and brief_file.stat().st_mtime > beats_file.stat().st_mtime
        video_format = state.get("format") or settings.get("format", "long")
        width, height = settings["video"][video_format]["width"], settings["video"][video_format]["height"]
        if beats_file.exists() and not state.get("manual") and not stale:
            found = [plain(b) for b in json.loads(beats_file.read_text(encoding="utf-8"))]
            print(f"  · beats: {len(found)} fragmentos de {beats_file.name} (bórralo para hacerlos de nuevo)")
            manual.write_beat_prompts(runs_dir(run_id), found, width, height, state.get("brief"))
            return {"beats": found}

        script = Script.model_validate(state["script"])
        audio = state["audio"]
        style = (state.get("brief") or {}).get("style") or options.get("image_style", "")
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
                    drafts |= {s.scene: s.ideas for s in draft.scenes if s.scene in wanted}

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
        result = [{k: b[k] for k in FIELDS if k in b} for b in result]
        beats_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        kinds = [b["kind"] for b in result]
        print(f"  · beats: {len(result)} fragmentos en {len(script.scenes)} escenas · "
              + ", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds)))
        manual.write_beat_prompts(runs_dir(run_id), result, width, height, state.get("brief"))
        return {"beats": result}

    return beats
