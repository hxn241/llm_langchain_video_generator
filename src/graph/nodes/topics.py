"""find_topics 🤖 + choose_topic ⏸: the topic of the video, when the run starts without --topic.

find_topics:  services/topics.py: the feeds of config.yaml (topics.feeds), the new entries scored
              by the model (llm.topics) and the topics not made yet (data/temas/registro.json).
              Saved as data/temas/<yyyymmdd-hhmm>.md (to read, with the sources of each one).
choose_topic: pauses the graph.
              approve -> the first one (the best score)
              edit    -> "elegido": another number
              reject  -> "x 3 5": those are discarded for good (status descartado) and the rest is shown
                         again; with any other text or Enter: other topics (the ones not shown yet,
                         after a new look at the feeds)
              The chosen one fills topic, urls (its articles), notes (its datazo), video_type and
              format (yours, if you gave --type or --format) and is recorded as en_curso.
"""
import re
from typing import Literal

from langgraph.types import Command

from llmkit import config as llm_config
from llmkit.approval import ask_approval
from services import topics as registry
from services.settings import load_settings

HOW_TO_CHOOSE = ("[y] el 1 · [e] cambia 'elegido' por otro número · [n] y \"x 3 5\": descarta esos para "
                 "siempre · [n] y Enter: otros temas")
DISCARD = re.compile(r"^\s*x\s+([\d\s,]+)$", re.I)


def make_find_topics(hub, settings=None):
    settings = settings or load_settings()

    def find_topics(state, config):
        path, items = registry.find(hub, skip=state.get("topics_seen", []))
        print(f"  · topics: {len(items)} temas en {llm_config.readable_path(path)}")
        return {"topic_candidates": items, "topics_seen": state.get("topics_seen", []) + [t["id"] for t in items]}

    return find_topics


def make_choose_topic():
    def choose_topic(state, config) -> Command[Literal["research", "find_topics", "choose_topic"]]:
        # Nothing with side effects before ask_approval: on resume this node runs again from the start
        items = state.get("topic_candidates") or []
        if not items:
            raise SystemExit("No hay temas nuevos: vuelve a probar otro día, añade feeds (topics.feeds en "
                             "config.yaml) o empieza con --topic")
        listing = "\n".join("         " + registry.short_line(n, t) for n, t in enumerate(items, 1))
        decision = ask_approval(
            "choose_topic",
            {"temas": "\n" + listing + "\n         " + HOW_TO_CHOOSE
                      + "\n         (cada uno con su datazo y sus fuentes en data/temas/, el .md más reciente)",
             "elegido": 1},
            description="Elige el tema del vídeo", allowed=("approve", "edit", "reject"))
        if decision["type"] == "reject":
            message = (decision.get("message") or "").strip()
            discard = DISCARD.match(message)
            if discard:
                numbers = {int(n) for n in re.findall(r"\d+", discard.group(1)) if 1 <= int(n) <= len(items)}
                registry.discard([items[n - 1]["id"] for n in numbers])
                print(f"  · choose_topic: {len(numbers)} descartados")
                rest = [t for n, t in enumerate(items, 1) if n not in numbers]
                return Command(goto="choose_topic" if rest else "find_topics", update={"topic_candidates": rest})
            return Command(goto="find_topics")
        chosen = items[0]
        if decision["type"] == "edit":
            picked = decision["edited_action"]["args"].get("elegido")
            if isinstance(picked, int) and 1 <= picked <= len(items):
                chosen = items[picked - 1]
            else:
                print(f"  · choose_topic: {picked!r} no es un número de 1 a {len(items)}: el 1")
        run_id = state.get("run_id") or config["configurable"]["thread_id"]
        registry.start_run(run_id, item=chosen)
        notes = "\n".join(n for n in (state.get("notes"), registry.notes_of(chosen)) if n)
        print(f"  · choose_topic: {chosen['topic']}")
        return Command(goto="research", update={
            "topic": chosen["topic"], "notes": notes, "topic_id": chosen["id"],
            "urls": list(dict.fromkeys((state.get("urls") or []) + [s["url"] for s in chosen["sources"]])),
            "video_type": state.get("video_type") or chosen["type"],
            "format": state.get("format") or chosen["format"]})

    return choose_topic
