"""LLM per node and LM Studio's VRAM.

The models come from the `llm` section of config.yaml: each node of the graph
inherits from `default` and overrides what it sets (provider, model,
temperature, max_tokens...). Keys stay in the .env (api_key_env).

    model = model_for("write_script")         # ChatOpenAI with that node's settings
    unload_local_models()                     # lms unload --all: frees the VRAM for Flux/Chatterbox

Every provider speaks the OpenAI protocol (LM Studio, OpenRouter...), so
ChatOpenAI is used with each provider's base_url.
"""
import os
import shutil
import subprocess
from pathlib import Path

import requests

from llmkit import config
from services.settings import CONFIG_FILE, load_settings  # noqa: F401  (CONFIG_FILE: used by LLMHub)

RESERVED_KEYS = {"providers", "default", "unload_after_text_phase"}   # in `llm`, everything else is a node


def _llm_settings(settings=None):
    llm = (settings or load_settings()).get("llm")
    if not llm or "default" not in llm or "providers" not in llm:
        raise ValueError("config.yaml needs an `llm` section with `providers` and `default`")
    return llm


def node_settings(node="default", settings=None):
    """`default` + the node's own settings. A node not listed uses `default`."""
    llm = _llm_settings(settings)
    if node in RESERVED_KEYS - {"default"}:
        raise ValueError(f"{node!r} is not a node")
    return {**llm["default"], **(llm.get(node) or {})}


def _provider(llm, name):
    if name not in llm["providers"]:
        raise ValueError(f"Unknown LLM provider {name!r}. Defined in config.yaml: {list(llm['providers'])}")
    provider = dict(llm["providers"][name])
    provider.pop("load", None)                      # how LM Studio loads it (prepare), not a client option
    if "api_key_env" in provider:
        variable = provider.pop("api_key_env")
        provider["api_key"] = os.getenv(variable)
        if not provider["api_key"]:
            raise RuntimeError(f"{variable} is missing in the .env file (needed by provider {name!r})")
    return provider


def model_for(node="default", settings=None, **overrides):
    """The chat model of `node` (see config.yaml). overrides: any option for this call only."""
    from langchain_openai import ChatOpenAI

    options = {**node_settings(node, settings), **overrides}
    llm = _llm_settings(settings)
    provider_name = options.pop("provider")
    provider = _provider(llm, provider_name)
    if provider_name == "local":
        _check_downloaded(options["model"], provider["base_url"])
    return ChatOpenAI(**{**provider, **options})          # node options win over the provider's


def prepare(node="default", settings=None):
    """Call right before using a node's model: if it is local, makes sure LM Studio has it
    loaded as config.yaml says (llm.providers.local.load). Otherwise LM Studio loads it
    on its own with ITS defaults, e.g. 111k of context: the VRAM overflows into system
    RAM and a script that takes 2 minutes takes more than 10."""
    options, llm = node_settings(node, settings), _llm_settings(settings)
    if options["provider"] != "local" or not llm["providers"]["local"].get("load"):
        return
    ensure_loaded(options["model"], llm["providers"]["local"]["base_url"], **llm["providers"]["local"]["load"])


def ensure_loaded(model, base_url, context_length, gpu="max", parallel=None, verbose=True):
    """Loads `model` in LM Studio with this context (and only it: one LLM at a time in 12 GB).
    Does nothing if it is already loaded like that, or if LM Studio cannot tell (not running:
    the request itself will then fail with a clear error)."""
    info = _model_info(model, base_url)
    if info is None or (info.get("state") == "loaded" and info.get("loaded_context_length") == context_length):
        return
    if verbose:
        print(f"  · LM Studio: loading {model} with {context_length} tokens of context...")
    _run_lms("unload", "--all")
    _run_lms("load", model, "--context-length", context_length, "--gpu", gpu, "-y",
             *(["--parallel", parallel] if parallel else []))


def _model_info(model, base_url):
    try:
        r = requests.get(f"{base_url.rstrip('/').removesuffix('/v1')}/api/v0/models/{model}", timeout=3)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def _run_lms(*args):
    result = subprocess.run([_lms(), *map(str, args)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"lms {' '.join(map(str, args))} failed: {(result.stderr or result.stdout).strip()}")


def local_models(base_url):
    """Ids of the models downloaded in LM Studio (loaded or not), or None if it cannot tell
    (not running, or a version without this API). /v1/models only lists the LOADED ones."""
    try:
        r = requests.get(f"{base_url.rstrip('/').removesuffix('/v1')}/api/v0/models", timeout=3)
        return {m["id"] for m in r.json()["data"]} if r.ok else None
    except (requests.RequestException, ValueError, KeyError):
        return None


def _check_downloaded(model, base_url):
    """LM Studio answers a request for a model it does NOT have with whatever model is
    loaded, without any error: config.yaml would say one model and another would run."""
    available = local_models(base_url)
    if available is not None and model not in available:
        raise ValueError(f"Model {model!r} (config.yaml) is not downloaded in LM Studio. "
                         f"Available: {sorted(available)}")


# ------------------------------------------------------------ LM Studio VRAM
def _lms():
    """Path of LM Studio's CLI (on the PATH, or where LM Studio installs it)."""
    found = shutil.which("lms") or Path.home() / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms")
    if not Path(found).exists():
        raise RuntimeError("LM Studio's `lms` CLI was not found. Install it from LM Studio "
                           "(Developer > Install lms CLI) or add ~/.lmstudio/bin to the PATH.")
    return str(found)


def lmstudio_running(settings=None):
    """True if LM Studio's server answers (if not, nothing of it is in the VRAM)."""
    base_url = _llm_settings(settings)["providers"].get("local", {}).get("base_url", config.LMSTUDIO_BASE_URL)
    try:
        return requests.get(f"{base_url.rstrip('/')}/models", timeout=3).ok
    except requests.RequestException:
        return False


def unload_local_models(settings=None, verbose=True):
    """Unloads every model from LM Studio (lms unload --all) to free the VRAM.

    Returns True if it unloaded, False if LM Studio was not running (nothing to free).
    The next call to a local model loads it again on its own (LM Studio's JIT loading),
    so it is safe to call even if a later node uses the LLM again (it only costs the load time).
    """
    if not lmstudio_running(settings):
        if verbose:
            print("  · LM Studio is not running: nothing to unload")
        return False
    _run_lms("unload", "--all")
    if verbose:
        print("  · LM Studio: models unloaded (VRAM free)")
    return True


def unload_after_text_phase(settings=None, verbose=True):
    """For the `unload_llm` node: unloads only if config.yaml says so (llm.unload_after_text_phase)."""
    llm = _llm_settings(settings)
    if not llm.get("unload_after_text_phase", False):
        return False
    return unload_local_models(settings, verbose)
