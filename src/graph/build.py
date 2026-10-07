"""Builds the LangGraph graph. It grows with the roadmap:

    research -> angles -> choose_angle ⏸ -> outline -> approve_outline ⏸ -> write_script -> unload_llm
                  ^_ rejected, with your reason _|      ^____ rejected, with your reason ____|
                                                          (--manual: -> manual_script ⏸, your script
                                                           and visuals in <run>/manual/ -> unload_llm)
        -> voice -> align -> beats -> visuals -> assemble -> END

    text phase (LLM):  research, angles, outline, write_script
    media phase:       voice (TTS) · align (Whisper) · beats (LLM) · visuals (SigLIP 2 + vision
                       LLM + HiDream) · assemble (ffmpeg). Each GPU step frees its VRAM when done.

Next (README): review after write_script, human_script_edit, subtitles, music, metadata...
"""
from langgraph.graph import END, START, StateGraph

from graph.nodes.align import make_align
from graph.nodes.angles import make_angles, make_choose_angle
from graph.nodes.assemble import make_assemble
from graph.nodes.beats import make_beats
from graph.nodes.manual_script import make_manual_script
from graph.nodes.outline import make_approve_outline, make_outline
from graph.nodes.research import make_research
from graph.nodes.unload_llm import unload_llm
from graph.nodes.visuals import make_visuals
from graph.nodes.voice import make_voice
from graph.nodes.write_script import make_write_script
from graph.state import VideoState

# Media nodes that can be run again on a run (scripts/run_pipeline.py --redo NODE):
# the node before each one, to make the graph continue from it.
REDO_AFTER = {"voice": "unload_llm", "align": "voice", "beats": "align", "visuals": "beats", "assemble": "visuals"}


def build_graph(hub, settings=None):
    """The compiled graph, with the hub's checkpointer (threads = runs, resumable)."""
    graph = StateGraph(VideoState)
    graph.add_node("research", make_research(hub, settings))
    graph.add_node("angles", make_angles(hub, settings))
    graph.add_node("choose_angle", make_choose_angle())                 # routes with Command
    graph.add_node("outline", make_outline(hub, settings))
    graph.add_node("approve_outline", make_approve_outline(settings))   # routes with Command
    graph.add_node("write_script", make_write_script(hub, settings))
    graph.add_node("manual_script", make_manual_script(settings))      # routes with Command
    graph.add_node("unload_llm", unload_llm)            # text phase -> media phase (VRAM)
    graph.add_node("voice", make_voice(settings))
    graph.add_node("align", make_align(settings))       # Whisper: the time of every word
    graph.add_node("beats", make_beats(hub, settings))  # LLM: what is shown, from which word
    graph.add_node("visuals", make_visuals(hub, settings))
    graph.add_node("assemble", make_assemble(settings))
    graph.add_edge(START, "research")
    graph.add_edge("research", "angles")
    graph.add_edge("angles", "choose_angle")
    graph.add_edge("outline", "approve_outline")
    graph.add_edge("write_script", "unload_llm")
    graph.add_edge("unload_llm", "voice")
    graph.add_edge("voice", "align")
    graph.add_edge("align", "beats")
    graph.add_edge("beats", "visuals")
    graph.add_edge("visuals", "assemble")
    graph.add_edge("assemble", END)
    return hub.workflow(graph)
