"""Shared terminal utilities: show what a graph does, ask for human approval
and turn API errors into short messages.

In a web application these functions would be the interface; the
LangChain/LangGraph logic (llmkit/*.py and the application) would not change.
"""
import json

from langgraph.types import Command

from llmkit.approval import approve_all, reject_all  # noqa: F401  (re-exported: they used to live here)


def describe_error(e):
    """API exception -> one readable line."""
    text = str(e)
    if "free-models-per-day" in text:
        return "OpenRouter's daily free-model limit is used up (it resets at 00:00 UTC)."
    if "429" in text:
        return "The model is saturated (429). Try again in a few minutes or change MODEL_NAME in the .env."
    if "503" in text or "overloaded" in text:
        return "The model provider is overloaded (503). Try again later."
    return f"{type(e).__name__}: {text[:200]}"


def run_cli(main):
    """Runs a program's main() with clean output: no huge tracebacks if the
    API fails, and Ctrl+C cancels quietly."""
    import sys

    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"\n❌ {describe_error(e)}")
        sys.exit(1)


def _message_summary(m):
    if getattr(m, "tool_calls", None):
        return "asks for " + ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
    return str(m.content).replace("\n", " ")[:160]


def run_graph(graph, input, config, verbose=True):
    """Runs (or resumes, with input=None or a Command) showing each node.

    Returns the interrupts the graph paused on ([] if it finished): usually one,
    several if parallel branches paused at the same time. Each has .value (the
    request) and .id. Works for any graph: message-based ones and ones with custom fields.
    """
    interrupts = {}
    for event in graph.stream(input, config, stream_mode="updates"):
        for node, changes in event.items():
            if node == "__interrupt__":
                interrupts.update({i.id: i for i in changes})
                continue
            if not verbose or not changes:
                continue
            if "messages" in changes:
                for m in changes["messages"]:
                    print(f"  · {node}: {_message_summary(m)}")
            else:
                fields = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in changes.items())
                print(f"  · {node}: {fields}")
    return list(interrupts.values())


def pending_interrupts(graph, config):
    """Requests of a graph that was left paused (e.g. when resuming another day)."""
    return [i.value for task in graph.get_state(config).tasks for i in task.interrupts]


def run_with_approvals(graph, input, config, decide, verbose=True):
    """Runs the graph and, every time it pauses, asks for the decision and resumes.

    decide(request) -> response: ask_decisions (a human), approve_all or
    reject_all (automatic, for evaluations). Returns True if it ever paused.
    """
    paused = False
    interrupts = run_graph(graph, input, config, verbose)
    while interrupts:
        paused = True
        if len(interrupts) == 1:
            resume = decide(interrupts[0].value)
        else:
            # Several branches paused at once: one answer each, by interrupt id
            resume = {i.id: decide(i.value) for i in interrupts}
        interrupts = run_graph(graph, Command(resume=resume), config, verbose)
    return paused


def _show(value):
    """Non-text values are shown as JSON, so they can be copied and edited."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _parse_like(text, original):
    """What a human typed -> a value of the same kind as `original`.
    Text stays text; anything else (numbers, lists, dicts...) is read as JSON,
    and if it is not valid JSON it is kept as text."""
    if isinstance(original, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def ask_decisions(request, read=input):
    """Shows the pending actions and asks what to do with each one.

    The format of llmkit/approval.py. "review_configs" says which decisions each action allows; if
    missing, only approve or reject.
    """
    allowed = {c["action_name"]: c["allowed_decisions"] for c in request.get("review_configs", [])}
    _read = read
    # Strips spaces and the invisible BOM that PowerShell adds with "y" | python ...
    read = lambda text: _read(text).strip().lstrip("﻿")  # noqa: E731
    decisions = []
    for action in request["action_requests"]:
        print(f"\n  ⚠️  Pending approval: {action['name']}")
        for key, value in action["args"].items():
            print(f"       {key}: {_show(value)}")
        can_edit = "edit" in allowed.get(action["name"], [])
        option = read("  [y]es / [n]o" + (" / [e]dit" if can_edit else "") + "? ").lower()
        if option == "y":
            decisions.append({"type": "approve"})
        elif option == "e" and can_edit:
            new_args = {}
            for key, value in action["args"].items():
                hint = "" if isinstance(value, str) else " (as JSON)"
                new_value = read(f"  New value for '{key}'{hint} (Enter = unchanged): ")
                new_args[key] = _parse_like(new_value, value) if new_value else value
            decisions.append({"type": "edit", "edited_action": {"name": action["name"], "args": new_args}})
        else:
            reason = read("  Reason for rejecting (optional): ")
            decisions.append({"type": "reject", "message": reason or "Rechazado por el usuario."})
    return {"decisions": decisions}
