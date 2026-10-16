"""labels 🤖: the on-screen labels (rótulos) -> state["labels"] and <run>/rotulos.json.

One LLM call (prompts/labels.md) reads the narrated script and the facts and decides, at most once
each, where to show:
    fuente    the study, journal or institution something comes from ("Universidad de Exeter",
              "Estudio publicado en Nature · 2026")
    persona   someone named ("Jolyon Troscianko", "Ecólogo visual · Universidad de Exeter")
    termino   a key concept, the first time it is explained ("Camuflaje de movimiento")
Each one starts on the word where it is said (its first words, found in the narration as beats does)
and assemble draws it over the video (graph/onscreen.py).

Portraits (video.labels.portraits): a persona label gets the person's photo ("foto", in
<run>/retratos/) when Wikidata has someone with exactly that name and a free photo on Commons
(services/portraits.py) AND a second LLM call (prompts/portraits.md, llm.portraits) says that card
is surely the person the video talks about. Anything that fails or is doubtful: no photo, the label
as text only. "foto_ficha" and "foto_quien" say where it came from, to check it at the pause.

<run>/rotulos.json can be edited by hand (change a text, delete one, delete a "foto"): if it exists
it is used as it is, only put on the current voice's times (a new voice: --redo voice); delete it
for new ones. video.labels.enabled: false -> no labels.
"""
import json

from graph.nodes.beats import locate
from graph.state import LabelsDraft, PortraitChoice, Script
from prompts import load_prompt
from services import portraits
from services.settings import load_settings, runs_dir

NODE = "labels"
PORTRAITS_NODE = "portraits"
FILE = "rotulos.json"
PORTRAITS_DIR = "retratos"
FIELDS = ("scene", "word", "start", "text", "kind", "title", "subtitle",
          "foto", "foto_ficha", "foto_quien", "foto_credito")
LEAD = 0.1                  # seconds before the word: the label is already moving in when it is said


def labels_request(state, script, video_format):
    facts = state.get("facts") or []
    parts = [f"Tema del vídeo: {state.get('topic') or script.title}", f"Título: {script.title}",
             f"Formato: {'short (algo más de un minuto): 1-3 rótulos' if video_format == 'short' else 'vídeo largo: 4-10 rótulos'}",
             "", "# Guion narrado (número de escena. narración)"]
    parts += [f"{s.id}. {s.narration}" for s in script.scenes]
    parts += ["", "# Hechos (de aquí salen los nombres, cargos, instituciones, revistas y años)"]
    parts += [f"{n}. {f['fact']}" for n, f in enumerate(facts, start=1)]
    return "\n".join(parts)


def place(label, script, audio):
    """A label (scene + its first words) -> word index and seconds in its scene's audio; None if its
    words are not in the scene's narration (a changed script)."""
    scene = next((s for s in script.scenes if s.id == label["scene"]), None)
    if scene is None:
        return None
    narration, words = scene.narration.split(), audio[scene.id]["words"]
    index = locate(label["text"], narration, -1)
    if index is None or index >= len(words):
        return None
    start = 0.0 if index == 0 else round(max(0.0, words[index]["start"] - LEAD), 3)
    return {**label, "word": index, "start": start}


def portrait_request(label, people, script, topic):
    scene = next((s for s in script.scenes if s.id == label["scene"]), None)
    parts = [f"Tema del vídeo: {topic}", "", "# El rótulo", f"{label['title']} · {label.get('subtitle') or '(sin cargo)'}",
             "", "# Donde se le nombra", scene.narration if scene else "", "", "# Fichas de Wikidata con ese nombre"]
    for p in people:
        years = (f"nació en {p['born'] or '?'}" + (f", murió en {p['died']}" if p["died"] else "")
                 if p["born"] or p["died"] else "años desconocidos")
        parts += [f"{p['id']}: {p['description'] or '(sin descripción)'} · {years}"
                  + (f" · ocupaciones: {', '.join(p['occupations'])}" if p["occupations"] else "")
                  + (f" · ha trabajado en: {', '.join(p['employers'])}" if p["employers"] else "")]
    return "\n".join(parts)


def portrait_for(label, script, topic, decide, folder):
    """The photo fields for a persona label, or {} (nobody with that name and a photo, not surely them,
    no free licence, any error)."""
    try:
        people = portraits.candidates(label["title"])
        if not people:
            print(f"  · labels: {label['title']}: sin foto en Wikidata")
            return {}
        choice = decide.invoke({"text": portrait_request(label, people, script, topic)})
        person = next((p for p in people if p["id"] == choice.id.strip()), None)
        if person is None:
            print(f"  · labels: {label['title']}: sin foto, no es seguro que sea la persona ({choice.reason[:160]})")
            return {}
        photo = portraits.download(person, folder / PORTRAITS_DIR)
        if photo is None:
            print(f"  · labels: {label['title']}: sin foto, la de Wikidata no tiene licencia libre")
            return {}
    except Exception as e:                                  # noqa: BLE001 - the label stays, as text
        print(f"  ⚠️  labels: {label['title']}: sin foto ({str(e).strip()[:200] or type(e).__name__})")
        return {}
    ficha = f"https://www.wikidata.org/wiki/{person['id']}"
    print(f"  · labels: retrato de {label['title']} ({person['description'] or 'sin descripción'}; {ficha})")
    return {"foto": photo["path"].relative_to(folder).as_posix(), "foto_ficha": ficha,
            "foto_quien": person["description"],
            "foto_credito": " · ".join(x for x in (photo["author"], photo["license"], photo["page_url"]) if x)}


def make_labels(hub, settings=None):
    settings = settings or load_settings()
    write = hub.chain(load_prompt(NODE), schema=LabelsDraft, node=NODE)
    decide = hub.chain(load_prompt(PORTRAITS_NODE), schema=PortraitChoice, node=PORTRAITS_NODE)

    def labels(state, config):
        options = settings["video"].get("labels") or {}
        if not options.get("enabled", True):
            return {"labels": []}
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        path = runs_dir(run_id) / FILE
        script, audio = Script.model_validate(state["script"]), state["audio"]
        video_format = state.get("format") or settings.get("format", "long")

        if path.exists():                       # yours / a previous run's: on the current voice
            saved = json.loads(path.read_text(encoding="utf-8"))
            found = [p for p in (place(l, script, audio) for l in saved) if p]
            print(f"  · labels: {len(found)} rótulos de {FILE} (bórralo para hacerlos de nuevo)"
                  + (f"; {len(saved) - len(found)} sin su texto en el guion" if len(found) < len(saved) else ""))
        else:
            draft = write.invoke({"text": labels_request(state, script, video_format)})
            seen, found = set(), []
            for d in draft.labels:
                key = (d.kind, d.title.strip().lower())
                placed = place({"scene": d.scene, "text": d.start, "kind": d.kind, "title": d.title.strip(),
                                "subtitle": d.subtitle.strip()}, script, audio)
                if placed and key not in seen:  # once each, and only where its words really are
                    seen.add(key)
                    found.append(placed)
            found.sort(key=lambda l: ([s.id for s in script.scenes].index(l["scene"]), l["start"]))
            kinds = [l["kind"] for l in found]
            print(f"  · labels: {len(found)} rótulos · "
                  + (", ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds)) or "ninguno")
                  + (f" ({len(draft.labels) - len(found)} descartados: repetidos o sin su texto)"
                     if len(found) < len(draft.labels) else ""))
            if options.get("portraits", True):
                topic = state.get("topic") or script.title
                found = [{**l, **portrait_for(l, script, topic, decide, path.parent)} if l["kind"] == "persona" else l
                         for l in found]
        found =[{k: l[k] for k in FIELDS if k in l} for l in found]
        path.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"labels": found}

    return labels
