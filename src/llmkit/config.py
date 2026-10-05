"""Central configuration: .env variables, project paths and models.

Everything that changes between your PC and production (keys, model, provider)
comes from the .env, never from the code.
"""
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# ---------------------------------------------------------------- console
# When the output goes to a file or a pipe (e.g. a scheduled task with a log),
# Windows writes in cp1252 and printing an emoji raises UnicodeEncodeError.
# UTF-8 always; anything that still cannot be written becomes "?" instead of failing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower().replace("-", "") != "utf8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------ .env and paths
ROOT = Path(__file__).resolve().parents[2]      # this project's folder (the one containing src/)

# The .env is searched from the folder you run from, walking up: that way
# ANOTHER project using llmkit has its own .env. If none is found, this
# project's .env is used (so it works wherever you run it from).
_ENV = find_dotenv(usecwd=True) or str(ROOT / ".env")
load_dotenv(_ENV)
BASE = Path(_ENV).resolve().parent              # the project that is using llmkit


def _folder(variable, default):
    """A path from the .env; if relative, it is taken from the .env's folder."""
    return (BASE / os.getenv(variable, default)).resolve()


DATA_DIR = _folder("DATA_DIR", "data")          # documents and evaluation cases (edited by hand)
STATE_DIR = _folder("STATE_DIR", ".state")      # what the programs generate: indexes, SQLite, reports


def readable_path(path):
    """Path relative to the project if inside it (shorter to print); otherwise absolute."""
    try:
        return str(Path(path).relative_to(BASE))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------- model
# Format: "provider:model". E.g.:
#   lmstudio:qwen/qwen3.5-9b                (on your PC with LM Studio: no limits, no cost)
#   openrouter:<vendor>/<model>             (cloud; needs OPENROUTER_API_KEY)
#   openai:gpt-4o-mini
MODEL_NAME = os.getenv("MODEL_NAME", "lmstudio:qwen/qwen3.5-9b")
# Model for the evaluation judges. Empty = same as MODEL_NAME.
# Better a more capable one than the evaluated model (a small judge gives inconsistent verdicts).
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "")
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "8"))   # anti-loop brake per turn
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# LM Studio speaks the OpenAI protocol. The URL is its local server.
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

# -------------------------------------------------------------- embeddings
# "lmstudio":   the embedding model loaded in LM Studio (texts do NOT leave your PC)
# "openrouter": cloud API (texts leave your PC)
# "local":      sentence-transformers inside Python (texts do NOT leave)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "lmstudio")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ------------------------------------------------------------ vector index
# "chroma": vector database on disk (.state/chroma). Only updates the chunks
#           that change. Needs: python -m pip install langchain-chroma
# "memory": InMemoryVectorStore, only in RAM while the program runs (rebuilt
#           every time). No dependencies; for tests and quick trials.
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")

# -------------------------------------------------------------- web search
# "duckduckgo": free, no key | "tavily": built for agents, needs TAVILY_API_KEY
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo")
SEARCH_REGION = os.getenv("SEARCH_REGION", "es-es")     # DuckDuckGo region: es-es, us-en, wt-wt (none)...

# Sent to Wikipedia and web pages (some sites reject requests without one)
USER_AGENT = os.getenv("USER_AGENT", "llmkit/1.0 (learning project)")


def _openrouter_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing in the .env file")
    return key


def create_model(temperature=0, full_name=None, **options):
    """Returns the chat model set in MODEL_NAME (or `full_name`).

    LM Studio and OpenRouter speak the OpenAI protocol, so ChatOpenAI is used
    with another URL. Any other provider goes through init_chat_model.
    options: anything else the model accepts (max_tokens=, timeout=, max_retries=...);
    they override the defaults below.
    """
    full_name = full_name or MODEL_NAME
    provider, _, name = full_name.partition(":")
    if provider == "lmstudio":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**{
            "model": name,
            "base_url": LMSTUDIO_BASE_URL,
            "api_key": "lm-studio",   # LM Studio needs no key, but the client requires one
            "temperature": temperature,
            "timeout": 300,           # local models can be slow
            "max_retries": 1,
            **options,
        })

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**{
            "model": name,
            "base_url": OPENROUTER_BASE_URL,
            "api_key": _openrouter_key(),
            "temperature": temperature,
            "timeout": 120,
            "max_retries": 3,         # free models sometimes return 429/503
            **options,
        })

    from langchain.chat_models import init_chat_model

    return init_chat_model(full_name, temperature=temperature, **options)


def create_judge_model(**options):
    """The model used by the evaluation judges (JUDGE_MODEL_NAME or, if empty,
    the same as MODEL_NAME)."""
    return create_model(full_name=JUDGE_MODEL_NAME or MODEL_NAME, **options)


def embedding_model_name(provider=None):
    """Name of the embedding model of `provider` (default: the one in the .env).
    Used to name the indexes: each model gets its own."""
    provider = provider or EMBEDDING_PROVIDER
    return {"local": LOCAL_EMBEDDING_MODEL, "lmstudio": LMSTUDIO_EMBEDDING_MODEL}.get(provider, EMBEDDING_MODEL)


def create_embeddings(provider=None):
    """Returns the embedding model (text -> vector).

    IMPORTANT: always index and query with the SAME model; vectors from
    different models are not comparable.
    """
    provider = provider or EMBEDDING_PROVIDER

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "For local embeddings install:  "
                "python -m pip install langchain-huggingface sentence-transformers"
            ) from e
        return HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
        )

    if provider == "lmstudio":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=LMSTUDIO_EMBEDDING_MODEL,
            base_url=LMSTUDIO_BASE_URL,
            api_key="lm-studio",
            check_embedding_ctx_length=False,   # send text, not OpenAI tokens
        )

    if provider == "openrouter":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=_openrouter_key(),
            # Without this LangChain sends OpenAI tokens instead of text, and
            # non-OpenAI models reject them.
            check_embedding_ctx_length=False,
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r} (use 'lmstudio', 'openrouter' or 'local')")
