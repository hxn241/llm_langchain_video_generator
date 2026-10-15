"""What a run spends: every call to a model (LLM or image) -> <run>/costes.jsonl, and the total per
node in <run>/costes.md.

    costs.start(run_folder)                          # from here on, calls are written to that run
    costs.record("research", model, usage=usage)     # an LLM call (usage: the API's, OpenAI format)
    costs.record("visuals_images", model, cost=0.07) # anything else that costs money
    costs.summary(run_folder)                        # -> costes.md (every call of the run, --resume too)

The cost is OpenRouter's own (usage.cost, asked for with usage.include in services/llm.py): exact,
with the cache discount. A provider that does not give it (LM Studio: free) leaves cost 0 and
only the tokens. LLMHub records every node's calls on its own (a callback on each model).
"""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

LOG, SUMMARY = "costes.jsonl", "costes.md"
_folder = None                # the run being recorded (None: calls are not written anywhere)


def start(folder):
    global _folder
    _folder = Path(folder)
    _folder.mkdir(parents=True, exist_ok=True)


def record(node, model, usage=None, cost=None):
    """One call. usage: {"prompt_tokens", "completion_tokens", "prompt_tokens_details": {"cached_tokens"},
    "cost"} as the API gives it; cost: the price when there are no tokens (an image)."""
    if _folder is None:
        return
    usage = usage or {}
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "node": node, "model": model,
             "input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0),
             "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
             "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
             "cost": float(cost if cost is not None else usage.get("cost") or 0)}
    with open(_folder / LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class UsageLog(BaseCallbackHandler):
    """On a chat model (callbacks=[UsageLog(node)]): records each of its answers."""

    def __init__(self, node):
        self.node = node

    def on_llm_end(self, response, **kwargs):
        output = response.llm_output or {}
        record(self.node, output.get("model_name", ""), usage=output.get("token_usage") or {})


def summary(folder):
    """costes.md from every call in costes.jsonl; returns the total line ("" if nothing was recorded)."""
    path = Path(folder) / LOG
    if not path.exists():
        return ""
    calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes = {}
    for c in calls:
        n = nodes.setdefault(c["node"], {"calls": 0, "input": 0, "output": 0, "cached": 0, "reasoning": 0,
                                         "cost": 0.0, "models": set()})
        n["calls"] += 1
        n["models"].add(c["model"])
        for key in ("input", "output", "cached", "reasoning", "cost"):
            n[key] += c.get(key, 0)                   # older lines have no "reasoning"
    total = sum(n["cost"] for n in nodes.values())
    rows = [f"| {node} | {', '.join(sorted(m for m in n['models'] if m)) or '-'} | {n['calls']} | "
            f"{n['input']:,} | {n['cached']:,} | {n['output']:,} | {n['reasoning']:,} | {n['cost']:.4f} |"
            for node, n in sorted(nodes.items(), key=lambda item: -item[1]["cost"])]
    text = "\n".join([
        f"# Costes del run ({len(calls)} llamadas, {total:.3f} $)", "",
        "De más a menos caro. Entrada: tokens enviados (cacheados: los cobrados al precio de caché). "
        "Salida: todo lo que escribe el modelo; razonamiento: la parte de la salida que piensa antes "
        "de responder (se cobra igual, no se ve).", "",
        "| Nodo | Modelo | Llamadas | Entrada | Cacheados | Salida | Razonamiento | $ |",
        "|---|---|---|---|---|---|---|---|", *rows,
        f"| **Total** | | {len(calls)} | | | | | **{total:.3f}** |", ""])
    (Path(folder) / SUMMARY).write_text(text, encoding="utf-8")
    return f"{total:.3f} $ en {len(calls)} llamadas"
