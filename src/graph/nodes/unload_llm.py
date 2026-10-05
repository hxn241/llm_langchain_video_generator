"""unload_llm: end of the text phase. Frees the LLM's VRAM (lms unload --all) before the
media phase (TTS, Flux...), which does not fit in 12 GB at the same time as the LLM.

Only if config.yaml says so (llm.unload_after_text_phase). If a later node uses
the LLM again, LM Studio loads it back on its own.
"""
from services import llm


def unload_llm(state):
    llm.unload_after_text_phase()
    return {}
