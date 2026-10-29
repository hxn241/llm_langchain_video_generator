"""visuals 🤖: one clip or image per beat -> state["visuals"][beat id].

For each beat (graph/nodes/beats.py: what must be seen, from which word):
1. Candidates from the sources of its kind (config.yaml: visuals.sources). Sources today:
       stock       Pexels / Pixabay videos and photos
       nasa        NASA photos (space, missions, the Earth from orbit; public domain)
       wikimedia   Wikimedia Commons photos of real, named things (free licences, with attribution)
   each searched with every query of the beat (most concrete first; the results interleaved), and
   the sources interleaved too, so each one gets its share of visuals.max_candidates. Searches are
   cached in <run>/visuals/searches.json. A kind without sources goes straight to step 3.
2. Ranking in two steps:
   a. SigLIP 2 (services/workers.py -> embed_worker.py) compares the beat (query + subject)
      with every candidate's thumbnail; the best visuals.top_k go on.
   b. The vision model (llm.visuals) sees them in ONE numbered grid, cropped like the frame,
      with the words being said, and picks one (prompts/visual_choice.md). Below
      visuals.min_score, none is used.
   Beats marked "generate" (beats decides: an object on a studio background, a period scene...)
   are not searched when visuals.direct_generate is on: they go straight to step 3.
3. Beats without a good candidate: their prompt is generated (visuals.generator): with a paid
   model through OpenRouter (services/image_api.py) or, locally, with Z-Image Turbo in ComfyUI
   (the LLM is unloaded first). If OpenRouter fails, the rest go to ComfyUI. Same prompt and
   settings -> the file is reused, not regenerated. The vision model then looks at each one
   (prompts/generated_check.md: deformities, text, what is said); a rejected one is made again
   with another seed (visuals.check_generated).

Long beats (visuals.photo_max_seconds: only long videos; shorts have no limit): a beat longer
than that is judged first among its VIDEOS only; if one is good it fills the beat. Otherwise a
photo never stays on screen longer than the limit: the beat gets more visuals that share its
time (state["visual_extras"], montaje/05b.jpg...): another good candidate or, failing that, its
prompt generated again with another framing.

Manual mode: beats of kind "manual" (your [file] marks in guion.md, graph/manual.py) are used as they are.
A clip or photo is used once per video. Files go to <run>/visuals/ (thumbs/ has the
candidates' previews; choices.json, what was picked for each beat and why). At the end, a copy
of each beat's visual goes to <run>/montaje/ in order (01.mp4, 02.jpg...: graph/montage.py),
which is what assemble uses after the review_visuals pause.
To add a source: a function (beat, orientation, options) -> candidates, in SOURCES.
"""
import hashlib
import io
import itertools
import json
import math
import re
from types import SimpleNamespace

from PIL import Image

from graph import manual, montage
from graph.nodes.beats import plain
from graph.state import GeneratedCheck, Script, VisualChoice
from prompts import load_prompt
from services import comfy, contact_sheet, costs, image_api, stock, workers
from services.settings import load_settings, runs_dir

NODE = "visuals"


# ---------------------------------------------------------------- sources
# Words that bring motion graphics, icons and cheap infographics from the stock banks, not footage
# (figures will be our own charts: the future `data` source). A query with one of them is dropped.
GRAPHIC_WORDS = {"chart", "charts", "graph", "graphs", "graphic", "infographic", "diagram", "icon", "icons",
                 "arrow", "arrows", "pie", "percent", "percentage", "symbol", "question",
                 "concept", "illustration", "animation", "3d", "render", "cartoon", "vector", "logo"}
# ...except in a visualization shot (an atom, a magnetic field): there a 3D animation is what is wanted
VISUALIZATION_WORDS = {"animation", "3d", "render"}
RENDERED = {"visualization", "cutaway"}       # registers (beats.json: look) where a render is what is wanted
# What the judge is told about each register (photo: everything else)
LOOK_TEXT = {
    "visualization": "VISUALIZACIÓN CIENTÍFICA (lo que se nombra no se puede fotografiar: vale un render o una "
                     "animación fotorrealista de eso mismo)",
    "cutaway": "VISTA POR DENTRO (corte, sección o carcasa transparente de algo real: vale un render fotorrealista "
               "o una foto real que enseñe su interior)",
    "analogy": "LA COMPARACIÓN del narrador (lo que se ve es lo comparado, no el tema del vídeo: foto o vídeo real)",
    "archive": "ESCENA DE ÉPOCA (foto o vídeo de archivo de esos años, o con su aspecto; nada moderno)",
}
GENERATED_RETRIES = 2         # a generated image the judge rejects is made again this many times (another seed)
SEARCH_VERSION = 3            # part of the cache key: bump it when what a search returns changes
                              # (3: candidates carry their source's caption, for the judge)
# A long beat's extra generated images: the same prompt with another framing, so they differ
REFRAMES = ["Another view of the same moment: a wide shot showing the whole place.",
            "Another view of the same moment: a close-up of one telling detail.",
            "Another view of the same moment: a medium shot from a different angle."]
LETTERS = "bcdefgh"           # montaje/05b, 05c...: a beat's extra files


def queries_of(beat):
    """The beat's stock searches, most concrete first (beats.json made before "queries": one "query"),
    without those that would bring graphics."""
    queries = [q for q in beat.get("queries") or [beat.get("query", "")] if q.strip()]
    banned = GRAPHIC_WORDS - VISUALIZATION_WORDS if beat.get("look") in RENDERED else GRAPHIC_WORDS
    return [q for q in queries if not banned & set(q.lower().replace("-", " ").split())]


def interleave(lists):
    """[a1, a2], [b1] -> [a1, b1, a2]."""
    return [x for group in itertools.zip_longest(*lists) for x in group if x]


def provider_candidates(name, providers):
    """A source made of providers of services/stock.py."""
    def candidates(beat, orientation, options, cache):
        """Every query of the beat, videos before photos; the results are interleaved query by query
        so the generic last query also gets a place among the first max_candidates."""
        per_query = []
        wanted = providers or options["stock_providers"]
        searched = ",".join(stock.available(wanted))     # a new key in the .env -> searched again
        for query in queries_of(beat):
            found = []
            for kind in ("video", "photo"):
                key = f"v{SEARCH_VERSION}|{searched}|{kind}|{orientation}|{options.get('per_page', 8)}|{query.lower()}"
                if key not in cache:
                    cache[key] = stock.search(query, kind, orientation, wanted, options.get("per_page", 8))
                found += [dict(c, query=query) for c in cache[key]]
            per_query.append(found)
        mixed = interleave(per_query)
        for c in mixed:
            c["source"], c["key"] = name, f"{c['provider']}_{c['id']}"
        return mixed
    return candidates


SOURCES = {"stock": provider_candidates("stock", None),           # visuals.stock_providers
           "nasa": provider_candidates("nasa", ["nasa"]),
           "wikimedia": provider_candidates("wikimedia", ["wikimedia"])}


def gather(beat, orientation, options, folder, cache, space=False):
    """Candidates of every source of the beat's kind, with their thumbnail on disk. NASA only in a video
    about space (brief: space): elsewhere its photos took the places of stock ("dry floor" -> Mars)."""
    if not queries_of(beat):
        return []
    per_source = []
    for name in options.get("sources", {}).get(beat["kind"], []):
        if name == "nasa" and not space:
            continue
        try:
            per_source.append(SOURCES[name](beat, orientation, options, cache))
        except Exception as e:                                   # one source failing does not stop the rest
            print(f"  · visuals: source {name!r} failed for beat {beat['id']}: {e}")
    found = {}
    for c in interleave(per_source):                             # each source gets its share of places
        found.setdefault(c["key"], c)
    ready = []
    for c in list(found.values())[:options.get("max_candidates", 12)]:
        try:
            c["thumb_path"] = str(stock.thumbnail(c, folder))
            ready.append(c)
        except Exception:
            continue
    return ready


# ----------------------------------------------------------------- choose
def choose(hub, beat, scene, candidates, aspect, brief=None):
    """The vision model sees the candidates in a grid -> (candidate or None, VisualChoice)."""
    sheet = contact_sheet.grid([open(c["thumb_path"], "rb").read() for c in candidates], aspect)
    choice = hub.look(load_prompt("visual_choice"), choice_text(beat, scene, candidates, brief), [sheet],
                      schema=VisualChoice, node=NODE)
    return (candidates[choice.best - 1] if 1 <= choice.best <= len(candidates) else None), choice


def choice_text(beat, scene, candidates, brief=None):
    """What the judge is told about the grid (also scripts/compare_judge.py, to test other models)."""
    labels = ", ".join(f"{n} {'vídeo' if c['kind'] == 'video' else 'foto'}" for n, c in enumerate(candidates, start=1))
    avoid = list(dict.fromkeys(beat.get("must_not_show", []) + (brief or {}).get("avoid", [])))
    return (f"Narración de la escena: \"{scene.narration}\"\n"
            f"Fragmento que se está diciendo mientras se ve la imagen: \"{beat['text']}\"\n"
            + (f"La idea que cuenta: {beat['idea']}\n" if beat.get("idea") else "")
            + f"Qué debería verse: {beat['subject']}\n"
            + f"Tipo de plano: {LOOK_TEXT.get(beat.get('look'), 'FOTO o VÍDEO REAL')}\n"
            + (f"Tiene que verse: {', '.join(beat['must_show'])}\n" if beat.get("must_show") else "")
            + (f"NO debe verse: {', '.join(avoid)}\n" if avoid else "")
            + captions_text(candidates)
            + f"Hay {len(candidates)} imágenes numeradas ({labels}). ¿Cuál cuenta mejor la idea del fragmento?")


def caption_of(candidate):
    """What the candidate's source says it is (services/stock.py); for one cached before captions, the
    words of its page's name (Pixabay, Pexels and Commons pages are named after the image)."""
    if candidate.get("caption"):
        return candidate["caption"]
    url = (candidate.get("page_url") or "").rstrip("/")
    if "nasa.gov" in url:
        return ""                                   # only an id
    name = url.split("/")[-1].removeprefix("File:")
    return re.sub(r"(\.\w+$)|(-?\d+$)", "", name).replace("-", " ").replace("_", " ").strip()


def captions_text(candidates):
    """The judge's lines with each image's caption ("" if none has one)."""
    lines = [f"{n}: {caption_of(c)[:220]}" for n, c in enumerate(candidates, start=1) if caption_of(c)]
    if not lines:
        return ""
    return ("Fichas de las imágenes (lo que dice su fuente que son; pueden faltar o estar en inglés):\n"
            + "\n".join(lines) + "\n")


def pick(hub, beat, scene, pool, aspect, brief, options, videos_first=False):
    """The vision model's choice among the best of `pool` -> (candidate or None, VisualChoice, good).
    videos_first: it first sees only the videos (a long beat is better filled by a clip)."""
    top_k, min_score = options.get("top_k", 8), options.get("min_score", 6)
    best = pool[:top_k]
    if videos_first:
        videos = [c for c in pool if c["kind"] == "video"][:top_k]
        if videos:
            picked, choice = choose(hub, beat, scene, videos, aspect, brief)
            good = picked is not None and choice.score >= min_score
            if good or {c["key"] for c in videos} == {c["key"] for c in best}:
                return picked, choice, good                    # the same grid again would be wasted
    picked, choice = choose(hub, beat, scene, best, aspect, brief)
    return picked, choice, picked is not None and choice.score >= min_score


def found_visual(picked, choice, path):
    return {"path": str(path), "kind": picked["kind"], "source": picked["source"],
            "provider": picked["provider"], "query": picked["query"], "score": choice.score,
            "source_url": picked["page_url"], "author": picked["author"], "license": picked.get("license"),
            "ai_generated": picked["ai_generated"],
            **({"duration": picked["duration"]} if picked["kind"] == "video" else {})}


def reused(old, beat, pool):
    """The judge's picks for this beat in an earlier try of the node (choices.json; 2026-10-25: a
    crash while generating judged everything again) -> the candidates again, [] if nothing fitted.
    None if the beat changed or a pick is no longer in its pool: then it is judged again."""
    if not old or old.get("text") != beat["text"] or old.get("subject") != beat["subject"]:
        return None
    by_url = {c["page_url"]: c for c in pool}
    urls = [old["picked"], *(e["picked"] for e in old.get("extras", []))] if old.get("used") else []
    if any(u not in by_url for u in urls):
        return None
    return [by_url[u] for u in urls]


def parts_of(beat, photo_max):
    """How many visuals a beat needs when it is not a video: 1, or more if a photo would stay
    longer than photo_max seconds (long videos only)."""
    if not photo_max or beat.get("seconds", 0) <= photo_max:
        return 1
    return math.ceil(beat["seconds"] / photo_max)


def write_credits(run_folder, chosen, extras=None):
    """<run>/creditos.txt: who made each image and video found, with its licence, to paste into the
    video's description (Wikimedia's CC BY / CC BY-SA require it; the rest are a courtesy). The
    portraits of the labels (rotulos.json, "foto_credito") too."""
    lines, seen = [], set()
    every = [v for beat_id in sorted(chosen) for v in [chosen[beat_id], *(extras or {}).get(beat_id, [])]]
    for v in every:
        if not v.get("source_url") or v["source_url"] in seen:
            continue
        seen.add(v["source_url"])
        lines.append(" · ".join(x for x in (v.get("author") or v["provider"], v.get("license"), v["source_url"]) if x))
    labels_file = run_folder / "rotulos.json"
    if labels_file.exists():
        lines += [c for c in dict.fromkeys(l.get("foto_credito") for l in json.loads(labels_file.read_text(encoding="utf-8")))
                  if c and c not in lines]
    (run_folder / "creditos.txt").write_text("Imágenes y vídeos:\n" + "\n".join(lines) + "\n", encoding="utf-8")


def seed_for(run_id, beat_id):
    """The same seed for a beat on every run of it: an unchanged prompt gives the same image."""
    return int(hashlib.md5(f"{run_id}-{beat_id}".encode()).hexdigest()[:8], 16)


def generate(beats, run_id, folder, width, height, settings, attempt=0):
    """The beats without a good candidate -> {beat id: generated visual}. visuals.generator:
    openrouter (paid, visuals.generate_model) or comfy (Z-Image, local). If OpenRouter fails (no
    credit, an error...), the rest go to ComfyUI. Same prompt and settings -> the file is reused.
    A long beat's extra image comes as the beat with "slot" (its key in the result: (id, n)) and
    "suffix" (b, c...: in its file name and seed). attempt > 0: another try of an image the judge
    rejected (another seed, another file)."""
    options = settings["visuals"]
    model, resolution = options.get("generate_model"), options.get("generate_resolution")
    paid = options.get("generator", "comfy") == "openrouter"
    out, spent, comfy_on, failed = {}, 0.0, False, ""
    again = f"-{attempt}" if attempt else ""
    try:
        for n, b in enumerate(beats, start=1):
            path = provider = None
            name = f"beat_{b['id']:03d}{b.get('suffix', '')}"
            if paid:
                key = hashlib.md5(f"{b['prompt']}|{model}|{resolution}|{width}x{height}{again}"
                                  .encode()).hexdigest()[:8]
                path = next(folder.glob(f"{name}_{key}.*"), None)
                provider = model
                if not path:
                    try:
                        path, cost = image_api.generate(b["prompt"], folder / f"{name}_{key}",
                                                        width, height, model, resolution)
                        spent += cost
                        costs.record("visuals_images", model, cost=cost)
                    except Exception as e:                    # no credit, still down after retries: the rest locally
                        failed = f"{model} failed ({str(e)[:200]}) and "
                        print(f"  ⚠️  visuals: {model} falló ({str(e)[:200]}); el resto se genera con ComfyUI")
                        paid, path = False, None
            if not path:
                if not comfy_on:
                    if not comfy.running(settings):
                        raise RuntimeError(f"{failed}{len(beats) - n + 1} fragments must be generated and ComfyUI is "
                                           "not running. Start it (the images already made are reused):\n"
                                           "  cd C:\\Users\\hexen\\ComfyUI; "
                                           ".venv\\Scripts\\python.exe main.py --listen 127.0.0.1 --port 8188")
                    comfy_on = True
                seed = seed_for(run_id, f"{b['id']}{b.get('suffix', '')}{again}")
                key = hashlib.md5(f"{b['prompt']}|{seed}|{width}x{height}|{settings.get('comfy', {})}"
                                  .encode()).hexdigest()[:8]
                path, provider = folder / f"{name}_{key}.png", "z-image-turbo"
                if not path.exists():
                    comfy.generate(b["prompt"], path, width, height, seed=seed, settings=settings)
            print(f"  · visuals: generada {n}/{len(beats)} (fragmento {b['id']}{b.get('suffix', '')}, "
                  f"{b['kind']}, {provider})")
            out[b.get("slot", b["id"])] = {"path": str(path), "kind": "generated", "source": "generate",
                                           "provider": provider, "prompt": b["prompt"], "ai_generated": True}
    finally:
        if comfy_on:
            comfy.free(settings)
        if spent:
            print(f"  · visuals: imágenes generadas en OpenRouter: {spent:.2f} $")
    return out


def preview(path, long_side=1024):
    """A generated image as a JPEG small enough for the judge, big enough to count heads and legs."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((long_side, long_side))
    data = io.BytesIO()
    image.save(data, "JPEG", quality=88)
    return data.getvalue()


def check_text(beat, scene, brief=None):
    """What the judge of a generated image is told (prompts/generated_check.md)."""
    avoid = list(dict.fromkeys(beat.get("must_not_show", []) + (brief or {}).get("avoid", [])))
    return (f"Narración de la escena: \"{scene.narration}\"\n"
            f"Fragmento que se dice mientras se ve: \"{beat['text']}\"\n"
            f"Qué debería verse: {beat['subject']}\n"
            f"Tipo de plano: {LOOK_TEXT.get(beat.get('look'), 'FOTO REAL')}\n"
            + (f"Tiene que verse: {', '.join(beat['must_show'])}\n" if beat.get("must_show") else "")
            + (f"NO debe verse: {', '.join(avoid)}\n" if avoid else "")
            + f"Lo que se le pidió al generador: {beat['prompt']}\n"
            "¿Se puede usar esta imagen en el vídeo?")


def review_generated(hub, made, planned, scenes, brief, generate_again):
    """Each generated image seen by the vision model (prompts/generated_check.md): one with a deformity,
    text or that does not show what is said is made again with another seed, up to GENERATED_RETRIES
    times (2026-10-23: cockroaches with two heads went straight into the video). One still wrong keeps
    its problems in "check" (lista.md shows them, to change at the review_visuals pause)."""
    by_slot = {b.get("slot", b["id"]): b for b in planned}
    pending = made
    for attempt in range(GENERATED_RETRIES + 1):
        failed = []
        for slot, visual in pending.items():
            b = by_slot[slot]
            try:
                verdict = hub.look(load_prompt("generated_check"), check_text(b, scenes[b["scene"]], brief),
                                   [preview(visual["path"])], schema=GeneratedCheck, node=NODE)
            except Exception as e:                # the judge down (429...): the image stays, marked to look at
                print(f"  ⚠️  visuals: no se pudo revisar la generada {b['id']}{b.get('suffix', '')} ({str(e)[:120]})")
                made[slot] = dict(visual, check=["sin revisar: el juez falló, mírala tú"])
                continue
            made[slot] = visual if verdict.usable else dict(visual, check=verdict.problems or [verdict.seen])
            if not verdict.usable:
                failed.append(b)
                print(f"  · visuals: generada {b['id']}{b.get('suffix', '')} rechazada "
                      f"({'; '.join(verdict.problems)[:160]})" + (" -> otra" if attempt < GENERATED_RETRIES else ""))
        if not failed or attempt == GENERATED_RETRIES:
            break
        pending = generate_again(failed, attempt + 1)
    wrong = [slot for slot, v in made.items() if v.get("check")]
    if wrong:
        print(f"  ⚠️  visuals: {len(wrong)} generadas siguen con fallos: revísalas en montaje/lista.md")
    return made


# ------------------------------------------------------------------- node
def make_visuals(hub, settings=None):
    settings = settings or load_settings()

    def visuals(state, config):
        options = settings["visuals"]
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        if state.get("keep_visuals") and state.get("visuals"):
            # run_pipeline --redo voice --keep-visuals: the same images on the new voice. beats kept
            # their ids (retime), so each chosen visual still goes with its beat: nothing searched,
            # judged or generated; montaje/ is written again with the new times
            chosen = {int(k): v for k, v in state["visuals"].items()}
            extras = {int(k): v for k, v in (state.get("visual_extras") or {}).items()}
            folder = montage.write(runs_dir(run_id), [plain(b) for b in state["beats"]], chosen, extras)
            write_credits(runs_dir(run_id), chosen, extras)          # the labels' portraits may be new
            print(f"  · visuals: las mismas imágenes con los tiempos nuevos, sin buscar ni generar ({folder})")
            return {"visuals": chosen, "visual_extras": extras, "keep_visuals": False}
        video_format = state.get("format") or settings.get("format", "long")
        width, height = settings["video"][video_format]["width"], settings["video"][video_format]["height"]
        orientation = "portrait" if height > width else "landscape"
        script = Script.model_validate(state["script"])
        scenes = {s.id: s for s in script.scenes}
        beats = [plain(b) for b in state["beats"]]           # not the old beat_images pause's files
        folder = runs_dir(run_id) / "visuals"
        (folder / "thumbs").mkdir(parents=True, exist_ok=True)

        # 1. candidates (searches cached in the run: a redo does not spend the providers' rate limits again)
        cache_file = folder / "searches.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        # beats marks what no bank will have (an object on a studio background, a period scene, a
        # visualization): with visuals.direct_generate those go straight to step 3, not searched
        direct = options.get("direct_generate", True)
        try:
            space = (state.get("brief") or {}).get("space", False)
            candidates = {b["id"]: [] if b["kind"] == "manual" or (direct and b.get("generate")) else
                          gather(b, orientation, options, folder / "thumbs", cache, space) for b in beats}
        finally:
            cache_file.write_text(json.dumps({k: v for k, v in cache.items() if v}, ensure_ascii=False),
                                  encoding="utf-8")       # empty results are not kept: maybe a provider failed

        # 2a. SigLIP 2: text-image similarity, every beat in one job (the model loads once)
        with_candidates = [b for b in beats if candidates[b["id"]]]
        if with_candidates:
            result = workers.run("embed", {"model": options.get("embed_model", "google/siglip2-base-patch16-224"),
                                           "items": [{"text": f"{queries_of(b)[0]}. {b['subject']}",
                                                      "images": [c["thumb_path"] for c in candidates[b["id"]]]}
                                                     for b in with_candidates]}, settings)
            for b, scores in zip(with_candidates, result["scores"]):
                for c, score in zip(candidates[b["id"]], scores):
                    c["similarity"] = score

        # 2b. the vision model picks among the best (yours, in manual mode, are already chosen)
        chosen = {b["id"]: {"path": b["path"], "kind": manual.kind_of(b["path"]), "source": "manual",
                            "provider": "manual", "ai_generated": False}
                  for b in beats if b["kind"] == "manual"}
        photo_max = (options.get("photo_max_seconds") or {}).get(video_format)   # long videos only
        brief, aspect = state.get("brief"), width / height
        used, log, extras = set(), {}, {}
        # the picks of an earlier try are kept (delete choices.json to have everything judged again)
        choices_file = folder / "choices.json"
        previous = json.loads(choices_file.read_text(encoding="utf-8")) if choices_file.exists() else {}
        for b in with_candidates:
            pool = sorted((c for c in candidates[b["id"]] if c["key"] not in used and c["similarity"] > -1),
                          key=lambda c: -c["similarity"])
            if not pool:
                continue
            old = previous.get(str(b["id"]))
            again = reused(old, b, pool)
            if again is not None:
                log[b["id"]] = old
                scores = [old["score"], *(e["score"] for e in old.get("extras", []))]
                for k, (c, score) in enumerate(zip(again, scores)):
                    used.add(c["key"])
                    visual = found_visual(c, SimpleNamespace(score=score), stock.download(c, width, height, folder))
                    if k:
                        extras.setdefault(b["id"], []).append(visual)
                    else:
                        chosen[b["id"]] = visual
                print(f"  · visuals: fragmento {b['id']} ({b['kind']}) -> como antes: "
                      + (f"{again[0]['source']} {again[0]['kind']} · {old['score']}/10" if again else "se genera"))
                continue
            parts = parts_of(b, photo_max)
            picked, choice, good = pick(hub, b, scenes[b["scene"]], pool, aspect, brief, options,
                                        videos_first=parts > 1)
            log[b["id"]] = {"text": b["text"], "idea": b.get("idea"), "subject": b["subject"], "queries": queries_of(b),
                            "query": picked["query"] if picked else None, "score": choice.score, "reason": choice.reason,
                            "seen": choice.seen,
                            "picked": picked["page_url"] if picked else None, "used": good,
                            **({"author": picked["author"], "license": picked.get("license")} if good else {})}
            print(f"  · visuals: fragmento {b['id']} ({b['kind']}) -> "
                  + (f"{picked['source']} {picked['kind']} · {choice.score}/10" if good else
                     f"nada encaja ({choice.score}/10) -> se genera"))
            if not good:
                continue
            used.add(picked["key"])
            chosen[b["id"]] = found_visual(picked, choice, stock.download(picked, width, height, folder))
            # a long beat filled by a photo: more good candidates share its time (the rest, generated below)
            for k in range(1, parts if picked["kind"] != "video" else 1):
                rest = [c for c in pool if c["key"] not in used]
                if not rest:
                    break
                more, more_choice, more_good = pick(hub, b, scenes[b["scene"]], rest, aspect, brief, options)
                if not more_good:
                    break
                used.add(more["key"])
                extras.setdefault(b["id"], []).append(
                    found_visual(more, more_choice, stock.download(more, width, height, folder)))
                log[b["id"]].setdefault("extras", []).append({"picked": more["page_url"], "score": more_choice.score})
                print(f"  · visuals: fragmento {b['id']}{LETTERS[k - 1]} -> {more['source']} {more['kind']} · "
                      f"{more_choice.score}/10")
        choices_file.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        write_credits(runs_dir(run_id), chosen, extras)
        hub.unload_local_models(verbose=False)                  # the LLM out: VRAM for Z-Image

        # 3. the rest: generated (and the extra images long beats still need, with another framing)
        rest = [b for b in beats if b["id"] not in chosen]
        for b in beats:
            if b["kind"] == "manual" or chosen.get(b["id"], {}).get("kind") == "video":
                continue
            have = len(extras.get(b["id"], []))
            rest += [dict(b, slot=(b["id"], k), suffix=LETTERS[k - 1],
                          prompt=f"{b['prompt']} {REFRAMES[(k - 1) % len(REFRAMES)]}")
                     for k in range(1 + have, parts_of(b, photo_max))]
        if rest:
            made = generate(rest, run_id, folder, width, height, settings)
            if options.get("check_generated", True):
                made = review_generated(hub, made, rest, scenes, brief,
                                        lambda failed, attempt: generate(failed, run_id, folder, width, height,
                                                                         settings, attempt))
            for slot, visual in made.items():
                if isinstance(slot, tuple):
                    extras.setdefault(slot[0], []).append(visual)
                else:
                    chosen[slot] = visual
        folder = montage.write(runs_dir(run_id), beats, chosen, extras)   # in order, to check before assemble
        print(f"  · visuals: todo en {folder} (lista.md: qué va en cada número)")
        return {"visuals": chosen, "visual_extras": extras, "visual_warnings": []}

    return visuals
