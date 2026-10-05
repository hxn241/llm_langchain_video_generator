"""The system prompts of the LLM nodes: one .md per node, in Spanish.

They are literal text (LLMHub.chain does not treat them as templates), so they
can contain braces or JSON examples. What changes per call (the news, the
targets, the feedback) goes in the human message, built by the node.
"""
from pathlib import Path

FOLDER = Path(__file__).parent


def load_prompt(name):
    """prompts/<name>.md"""
    return (FOLDER / f"{name}.md").read_text(encoding="utf-8").strip()
