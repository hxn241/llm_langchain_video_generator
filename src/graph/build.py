"""Builds the LangGraph graph. It grows with the roadmap:

    research -> angles -> choose_angle ⏸ -> outline -> approve_outline ⏸ -> write_script -> choose_script ⏸
                  ^_ rejected, with your reason _|      ^____ rejected, with your reason ____|
        long videos (research.deep): approve_outline -> deep_research (the context of each section:
                       web + Wikipedia) -> review_research ⏸ (facts per section, how long the video
                       comes out; [n] your links -> deep_research again) -> write_script
        choose_script: [y] this script -> unload_llm
                       [n] yours -> manual_script ⏸ (guion.md + your visuals in <run>/manual/) -> unload_llm
                       (--manual: straight from the outline to manual_script, without write_script)
        -> voice -> align -> brief -> beats -> visuals -> review_visuals ⏸ (everything in order in
        <run>/montaje/, to replace, remove or add files; no judge) -> assemble -> END

    text phase (LLM):  research, angles, outline, write_script
    media phase:       voice (TTS) · align (Whisper) · brief + beats (LLM: the visual plan) · visuals
                       (SigLIP 2 + vision LLM + Z-Image) · assemble (ffmpeg). Each GPU step frees its
                       VRAM when done.

Next (README): review after write_script, human_script_edit, music, metadata...
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
from graph.nodes.review_visuals import make_review_visuals
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
    graph.add_node("research", make_research(hub, settings))
    graph.add_node("angles", make_angles(hub, settings))
    graph.add_node("choose_angle", make_choose_angle())                 # routes with Command
    graph.add_node("outline", make_outline(hub, settings))
    graph.add_node("approve_outline", make_approve_outline(settings))   # routes with Command
    graph.add_node("deep_research", make_deep_research(hub, settings))  # long videos: each section's context
    graph.add_node("review_research", make_review_research(settings))   # routes with Command
    graph.add_node("write_script", make_write_script(hub, settings))
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
    graph.add_edge(START, "research")
    graph.add_edge("research", "angles")
    graph.add_edge("angles", "choose_angle")
    graph.add_edge("outline", "approve_outline")
    graph.add_edge("deep_research", "review_research")
    graph.add_edge("write_script", "choose_script")
    graph.add_edge("unload_llm", "voice")
    graph.add_edge("voice", "align")
    graph.add_edge("align", "brief")
    graph.add_edge("brief", "beats")
    graph.add_edge("beats", "visuals")
    graph.add_edge("visuals", "review_visuals")
    graph.add_edge("assemble", END)
    return hub.workflow(graph)
