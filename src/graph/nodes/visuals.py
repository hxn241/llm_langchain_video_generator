"""visuals 🤖: one clip or image per beat -> state["visuals"][beat id].

For each beat (graph/nodes/beats.py: what must be seen, from which word):
1. Candidates from the sources of its kind (config.yaml: visuals.sources). Sources today:
       stock   Pexels / Pixabay videos and photos, searched with each of the beat's queries
               (most concrete first; the results interleaved). Searches are cached in
               <run>/visuals/searches.json.
   A kind without sources goes straight to step 3.
2. Ranking in two steps:
   a. SigLIP 2 (services/workers.py -> embed_worker.py) compares the beat (query + subject)
      with every candidate's thumbnail; the best visuals.top_k go on.
   b. The vision model (llm.visuals) sees them in ONE numbered grid, cropped like the frame,
      with the words being said, and picks one (prompts/visual_choice.md). Below
      visuals.min_score, none is used.
3. Beats without a good candidate: the LLM is unloaded and HiDream-O1 (ComfyUI) generates
   the beat's prompt. Same prompt + same seed -> the file is reused, not regenerated.

Manual mode: beats of kind "manual" (your files, graph/manual.py) are used as they are.
A clip or photo is used once per video. Files go to <run>/visuals/ (thumbs/ has the
candidates' previews; choices.json, what was picked for each beat and why).
To add a source: a function (beat, orientation, options) -> candidates, in SOURCES.
"""
import hashlib
import itertools
import json

from graph import manual
from graph.state import Script, VisualChoice
from prompts import load_prompt
from services import comfy, contact_sheet, stock, workers
from services.settings import load_settings, runs_dir

NODE = "visuals"


# ---------------------------------------------------------------- sources
# Words that bring motion graphics, icons and cheap infographics from the stock banks, not footage
# (figures will be our own charts: the future `data` source). A query with one of them is dropped.
GRAPHIC_WORDS = {"chart", "charts", "graph", "graphs", "graphic", "infographic", "diagram", "icon", "icons",
                 "arrow", "arrows", "pie", "percent", "percentage", "symbol", "question",
                 "concept", "illustration", "animation", "3d", "render", "cartoon", "vector", "logo"}
SEARCH_VERSION = 2            # part of the cache key: bump it when what a search returns changes


def queries_of(beat):
    """The beat's stock searches, most concrete first (beats.json made before "queries": one "query"),
    without those that would bring graphics."""
    queries = [q for q in beat.get("queries") or [beat.get("query", "")] if q.strip()]
    return [q for q in queries if not GRAPHIC_WORDS & set(q.lower().replace("-", " ").split())]


def stock_candidates(beat, orientation, options, cache):
    """Every query of the beat, videos before photos; the results are interleaved query by query
    so the generic last query also gets a place among the first max_candidates."""
    per_query = []
    providers = ",".join(stock.available(options["stock_providers"]))   # a new key in the .env -> searched again
    for query in queries_of(beat):
        found = []
        for kind in ("video", "photo"):
            key = f"v{SEARCH_VERSION}|{providers}|{kind}|{orientation}|{options.get('per_page', 8)}|{query.lower()}"
            if key not in cache:
                cache[key] = stock.search(query, kind, orientation, options["stock_providers"],
                                          options.get("per_page", 8))
            found += [dict(c, query=query) for c in cache[key]]
        per_query.append(found)
    mixed = [c for group in itertools.zip_longest(*per_query) for c in group if c]
    for c in mixed:
        c["source"], c["key"] = "stock", f"{c['provider']}_{c['id']}"
    return mixed


SOURCES = {"stock": stock_candidates}


def gather(beat, orientation, options, folder, cache):
    """Candidates of every source of the beat's kind, with their thumbnail on disk."""
    if not queries_of(beat):
        return []
    found = {}
    for name in options.get("sources", {}).get(beat["kind"], []):
        try:
            found |= {c["key"]: c for c in SOURCES[name](beat, orientation, options, cache) if c["key"] not in found}
        except Exception as e:                                   # one source failing does not stop the rest
            print(f"  · visuals: source {name!r} failed for beat {beat['id']}: {e}")
    ready = []
    for c in list(found.values())[:options.get("max_candidates", 12)]:
        try:
            c["thumb_path"] = str(stock.thumbnail(c, folder))
            ready.append(c)
        except Exception:
            continue
    return ready


# ----------------------------------------------------------------- choose
def choose(hub, beat, scene, candidates, aspect):
    """The vision model sees the candidates in a grid -> (candidate or None, VisualChoice)."""
    labels = ", ".join(f"{n} {'vídeo' if c['kind'] == 'video' else 'foto'}" for n, c in enumerate(candidates, start=1))
    text = (f"Narración de la escena: \"{scene.narration}\"\n"
            f"Fragmento que se está diciendo mientras se ve la imagen: \"{beat['text']}\"\n"
            f"Qué debería verse: {beat['subject']}\n"
            f"Hay {len(candidates)} imágenes numeradas ({labels}). ¿Cuál muestra mejor lo que se dice en el fragmento?")
    sheet = contact_sheet.grid([open(c["thumb_path"], "rb").read() for c in candidates], aspect)
    choice = hub.look(load_prompt("visual_choice"), text, [sheet], schema=VisualChoice, node=NODE)
    return (candidates[choice.best - 1] if 1 <= choice.best <= len(candidates) else None), choice


def seed_for(run_id, beat_id):
    """The same seed for a beat on every run of it: an unchanged prompt gives the same image."""
    return int(hashlib.md5(f"{run_id}-{beat_id}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------- node
def make_visuals(hub, settings=None):
    settings = settings or load_settings()

    def visuals(state, config):
        options = settings["visuals"]
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        video_format = state.get("format") or settings.get("format", "long")
        width, height = settings["video"][video_format]["width"], settings["video"][video_format]["height"]
        orientation = "portrait" if height > width else "landscape"
        script = Script.model_validate(state["script"])
        scenes = {s.id: s for s in script.scenes}
        beats = state["beats"]
        folder = runs_dir(run_id) / "visuals"
        (folder / "thumbs").mkdir(parents=True, exist_ok=True)

        # 1. candidates (searches cached in the run: a redo does not spend the providers' rate limits again)
        cache_file = folder / "searches.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        try:
            candidates = {b["id"]: gather(b, orientation, options, folder / "thumbs", cache) for b in beats}
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
        used, log = set(), {}
        for b in with_candidates:
            best = sorted((c for c in candidates[b["id"]] if c["key"] not in used and c["similarity"] > -1),
                          key=lambda c: -c["similarity"])[:options.get("top_k", 8)]
            if not best:
                continue
            picked, choice = choose(hub, b, scenes[b["scene"]], best, width / height)
            good = picked is not None and choice.score >= options.get("min_score", 6)
            log[b["id"]] = {"text": b["text"], "subject": b["subject"], "queries": queries_of(b),
                            "query": picked["query"] if picked else None, "score": choice.score, "reason": choice.reason,
                            "picked": picked["page_url"] if picked else None, "used": good}
            print(f"  · visuals: fragmento {b['id']} ({b['kind']}) -> "
                  + (f"{picked['source']} {picked['kind']} · {choice.score}/10" if good else
                     f"nada encaja ({choice.score}/10) -> se genera"))
            if good:
                path = stock.download(picked, width, height, folder)
                used.add(picked["key"])
                chosen[b["id"]] = {"path": str(path), "kind": picked["kind"], "source": picked["source"],
                                   "provider": picked["provider"], "query": picked["query"], "score": choice.score,
                                   "source_url": picked["page_url"], "author": picked["author"],
                                   "ai_generated": picked["ai_generated"],
                                   **({"duration": picked["duration"]} if picked["kind"] == "video" else {})}
        (folder / "choices.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        hub.unload_local_models(verbose=False)                  # the LLM out: VRAM for HiDream

        # 3. the rest: generated
        rest = [b for b in beats if b["id"] not in chosen]
        if rest:
            if not comfy.running(settings):
                raise RuntimeError(f"{len(rest)} fragments must be generated and ComfyUI is not running. Start it:\n"
                                   "  cd C:\\Users\\hexen\\ComfyUI; .venv\\Scripts\\python.exe main.py "
                                   "--listen 127.0.0.1 --port 8188")
            try:
                for n, b in enumerate(rest, start=1):
                    seed = seed_for(run_id, b["id"])
                    key = hashlib.md5(f"{b['prompt']}|{seed}|{width}x{height}|{settings.get('comfy', {})}"
                                      .encode()).hexdigest()[:8]
                    path = folder / f"beat_{b['id']:03d}_{key}.png"
                    if not path.exists():
                        comfy.generate(b["prompt"], path, width, height, seed=seed, settings=settings)
                    print(f"  · visuals: generada {n}/{len(rest)} (fragmento {b['id']}, {b['kind']})")
                    chosen[b["id"]] = {"path": str(path), "kind": "generated", "source": "generate",
                                       "provider": "hidream-o1", "prompt": b["prompt"], "ai_generated": True}
            finally:
                comfy.free(settings)
        return {"visuals": chosen, "visual_warnings": []}

    return visuals
