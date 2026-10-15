"""Builds the LangGraph graph. It grows with the roadmap:

    (without --topic) find_topics -> choose_topic ⏸ -> research   (the topics of the feeds; [n] others)
    research -> angles -> choose_angle ⏸ -> outline -> approve_outline ⏸ -> write_script -> review_script -> choose_script ⏸
                  ^_ rejected, with your reason _|      ^____ rejected, with your reason ____|
        research: news or explainer (--type, or the LLM decides): news searches, or the general web
                       + Wikipedia
        research.deep (per format) and some section with questions: approve_outline -> deep_research
                       (the context of each section: web + Wikipedia) -> review_research ⏸ (facts per
                       section, how long the video comes out; [n] your links -> deep_research again)
                       -> write_script
        review_script: serious problems -> write_script again with its notes (review.max_revisions)
        choose_script: [y] this script -> unload_llm
                       [n] + your reason -> write_script rewrites it (and review_script again)
                       [n] + Enter: yours -> manual_script ⏸ (guion.md + your visuals in <run>/manual/) -> unload_llm
                       (--manual: straight from the outline to manual_script, without write_script)
        -> voice -> align -> brief -> beats -> visuals -> review_visuals ⏸ (everything in order in
        <run>/montaje/, to replace, remove or add files; no judge) -> assemble -> END

    text phase (LLM):  find_topics, research, angles, outline, deep_research, write_script, review_script
    media phase:       voice (TTS) · align (Whisper) · brief + beats (LLM: the visual plan) · visuals
                       (SigLIP 2 + vision LLM + Z-Image) · assemble (ffmpeg). Each GPU step frees its
                       VRAM when done.

"""
from langgraph.graph import END, START, StateGraph

from graph.nodes.align import make_align
from graph.nodes.angles import make_angles, make_choose_angle
from graph.nodes.assemble import make_assemble
from graph.nodes.beats import make_beats
from graph.nodes.brief import make_brief
from graph.nodes.deep_research import make_deep_research, make_review_research
from graph.nodes.manual_script import make_choose_script, make_manual_script
from graph.nodes.outline import make_approve_outline, make_outline
from graph.nodes.research import make_research
from graph.nodes.review_script import make_review_script
from graph.nodes.review_visuals import make_review_visuals
from graph.nodes.topics import make_choose_topic, make_find_topics
from graph.nodes.unload_llm import unload_llm
from graph.nodes.visuals import make_visuals
from graph.nodes.voice import make_voice
from graph.nodes.write_script import make_write_script
from graph.state import VideoState

# Media nodes that can be run again on a run (scripts/run_pipeline.py --redo NODE):
# the node before each one, to make the graph continue from it.
REDO_AFTER = {"voice": "unload_llm", "align": "voice", "brief": "align", "beats": "brief", "visuals": "beats",
              "assemble": "visuals"}


def build_graph(hub, settings=None):
    """The compiled graph, with the hub's checkpointer (threads = runs, resumable)."""
    graph = StateGraph(VideoState)
    graph.add_node("find_topics", make_find_topics(hub, settings))   # without --topic: the feeds' topics
    graph.add_node("choose_topic", make_choose_topic())                # routes with Command
    graph.add_node("research", make_research(hub, settings))
    graph.add_node("angles", make_angles(hub, settings))
    graph.add_node("choose_angle", make_choose_angle())                 # routes with Command
    graph.add_node("outline", make_outline(hub, settings))
    graph.add_node("approve_outline", make_approve_outline(settings))   # routes with Command
    graph.add_node("deep_research", make_deep_research(hub, settings))  # long videos: each section's context
    graph.add_node("review_research", make_review_research(settings))   # routes with Command
    graph.add_node("write_script", make_write_script(hub, settings))
    graph.add_node("review_script", make_review_script(hub, settings))  # routes with Command
    graph.add_node("choose_script", make_choose_script())              # routes with Command
    graph.add_node("manual_script", make_manual_script(settings))      # routes with Command
    graph.add_node("unload_llm", unload_llm)            # text phase -> media phase (VRAM)
    graph.add_node("voice", make_voice(settings))
    graph.add_node("align", make_align(settings))       # Whisper: the time of every word
    graph.add_node("brief", make_brief(hub, settings))  # LLM: the visual universe of the video
    graph.add_node("beats", make_beats(hub, settings))  # LLM: what is shown, from which word
    graph.add_node("visuals", make_visuals(hub, settings))
    graph.add_node("review_visuals", make_review_visuals())            # routes with Command
    graph.add_node("assemble", make_assemble(settings))
    graph.add_conditional_edges(START, lambda state: "research" if state.get("topic") else "find_topics",
                                ["research", "find_topics"])
    graph.add_edge("find_topics", "choose_topic")
    graph.add_edge("research", "angles")
    graph.add_edge("angles", "choose_angle")
    graph.add_edge("outline", "approve_outline")
    graph.add_edge("deep_research", "review_research")
    graph.add_edge("write_script", "review_script")
    graph.add_edge("unload_llm", "voice")
    graph.add_edge("voice", "align")
    graph.add_edge("align", "brief")
    graph.add_edge("brief", "beats")
    graph.add_edge("beats", "visuals")
    graph.add_edge("visuals", "review_visuals")
    graph.add_edge("assemble", END)
    return hub.workflow(graph)
