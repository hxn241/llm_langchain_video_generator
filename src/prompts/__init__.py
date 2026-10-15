"""The system prompts of the LLM nodes: one .md per node, in Spanish.

They are literal text (LLMHub.chain does not treat them as templates), so they
can contain braces or JSON examples. What changes per call (the news, the
targets, the feedback) goes in the human message, built by the node.

Only two marks are replaced, with the channel of config.yaml (channel:): {{canal}} (who the
channel is, in one sentence) and {{temas}} (what its videos are about). Another channel is
another config, with the same prompts.
"""
from pathlib import Path

from services.settings import load_settings

FOLDER = Path(__file__).parent


def load_prompt(name, settings=None, folder=None):
    """prompts/<name>.md, with the channel in it. folder: another folder of prompts (a test version)."""
    channel = (settings or load_settings()).get("channel") or {}
    text = (Path(folder or FOLDER) / f"{name}.md").read_text(encoding="utf-8").strip()
    return (text.replace("{{canal}}", channel.get("description", "un canal de YouTube en español de divulgación"))
                .replace("{{temas}}", channel.get("topics", "ciencia y tecnología")))
