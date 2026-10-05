"""Human approval: ONE format for every pause, in agents and in workflows.

It is the format of create_agent's HumanInTheLoopMiddleware, so the same
console (console.ask_decisions) and the same automatic deciders (approve_all,
reject_all) work with create_agent, llmkit/agent.py and your own workflows.

    pauses with:   {"action_requests": [{"name", "args", "description"}, ...],
                    "review_configs":  [{"action_name", "allowed_decisions"}, ...]}
    resumes with:  Command(resume={"decisions": [
                       {"type": "approve"}
                     | {"type": "edit", "edited_action": {"name": ..., "args": {...}}}
                     | {"type": "reject", "message": "..."}, ...]})
                   (one decision per action_request, in the same order)

In a workflow node:

    decision = ask_approval("approve_plan", {"plan": state["plan"]}, "Review the outline")
    if decision["type"] == "reject": ...
"""
from langgraph.types import interrupt

DECISIONS = ("approve", "edit", "reject")


def approval_request(actions, allowed=DECISIONS):
    """actions: [{"name", "args", "description"?}, ...] -> the pause request."""
    return {
        "action_requests": [{"name": a["name"], "args": a["args"],
                             "description": a.get("description", "Requires approval")} for a in actions],
        "review_configs": [{"action_name": a["name"], "allowed_decisions": list(allowed)} for a in actions],
    }


def ask_approval(name, args, description="Requires approval", allowed=DECISIONS):
    """Pauses the graph (interrupt) until a human decides one action; returns the decision.

    CAREFUL: on resume LangGraph runs the node AGAIN from the start, so put
    nothing with side effects before this call. Needs a checkpointer.
    """
    response = interrupt(approval_request([{"name": name, "args": args, "description": description}], allowed))
    return response["decisions"][0]


def interrupt_on(tool_names, allowed=DECISIONS):
    """The `interrupt_on` of create_agent's HumanInTheLoopMiddleware for these tools."""
    return {name: {"allowed_decisions": list(allowed)} for name in tool_names}


# ------------------------------------------------- automatic deciders (evaluations)
def approve_all(request):
    return {"decisions": [{"type": "approve"} for _ in request["action_requests"]]}


def reject_all(request):
    return {"decisions": [{"type": "reject", "message": "Rechazado automáticamente (evaluación)."}
                          for _ in request["action_requests"]]}
