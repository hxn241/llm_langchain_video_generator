"""Agent with tools and human approval, as a LangGraph graph.

    START -> model --asks for tools?--> approval -> tools -> model ...
             model --no----------------------------------------> END
             model --LLM call limit for this turn--> limit -----> END

It is the "hand-made" version of LangChain's create_agent. Use it when you
need to control every step; otherwise use create_agent (LLMHub.agent does).

Human approval: the `approval` node calls interrupt() ONLY if some tool is
sensitive. It uses the format of llmkit/approval.py (the same as create_agent's
HumanInTheLoopMiddleware), so the console works for both variants.
"""
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from llmkit.approval import approval_request


class AgentState(MessagesState):
    """Only `messages`, with the add_messages reducer: appends new messages and
    REPLACES those whose id already exists (which is what allows editing a tool call)."""


def llm_calls_this_turn(messages):
    """Model replies since the user's last message."""
    n = 0
    for m in reversed(messages):
        if m.type == "human":
            break
        n += m.type == "ai"
    return n


def create_agent_graph(model, tools, system_prompt, sensitive_tools=frozenset(),
                       checkpointer=None, max_llm_calls=8):
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    # ------------------------------------------------------------------ nodes
    def model_node(state):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        return {"messages": [model_with_tools.invoke(messages)]}

    def approval_node(state) -> Command[Literal["tools", "model"]]:
        """Pauses if any requested tool is sensitive.

        CAREFUL: on resume, LangGraph runs this node AGAIN from the start
        (interrupt then returns the answer). Nothing with side effects before interrupt().
        """
        calls = state["messages"][-1].tool_calls
        sensitive = [c for c in calls if c["name"] in sensitive_tools]
        if not sensitive:
            return Command(goto="tools")

        response = interrupt(approval_request(sensitive))
        decisions = {c["id"]: d for c, d in zip(sensitive, response["decisions"])}

        rejected = [c for c in sensitive if decisions[c["id"]]["type"] == "reject"]
        if rejected:
            # If anything is rejected, no tool runs, and every tool_call needs
            # its ToolMessage reply (otherwise the model API returns an error).
            # The text goes to the model, so it is in Spanish like the prompts.
            reason = decisions[rejected[0]["id"]].get("message", "")
            return Command(goto="model", update={"messages": [
                ToolMessage(content=f"El usuario RECHAZÓ esta acción. {reason} No la repitas; "
                                    "propón una alternativa.",
                            tool_call_id=c["id"], name=c["name"], status="error")
                for c in calls
            ]})

        edited = {i: d["edited_action"] for i, d in decisions.items() if d["type"] == "edit"}
        if edited:
            # Editing = replacing the model's message with a copy with the new
            # args. Same id -> add_messages REPLACES it.
            last = state["messages"][-1]
            new_calls = [{**c, **edited[c["id"]]} if c["id"] in edited else c for c in last.tool_calls]
            return Command(goto="tools",
                           update={"messages": [last.model_copy(update={"tool_calls": new_calls})]})
        return Command(goto="tools")

    def tools_node(state):
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:  # the LLM may make up names
                content = f"La herramienta '{call['name']}' no existe. Disponibles: {list(tools_by_name)}"
                status = "error"
            else:
                try:
                    content, status = tool.invoke(call["args"]), "success"
                except Exception as e:  # a failing tool does not bring down the graph
                    content, status = f"Error ejecutando la herramienta: {e}", "error"
            results.append(ToolMessage(content=str(content), tool_call_id=call["id"],
                                       name=call["name"], status=status))
        return {"messages": results}

    def limit_node(state):
        """Closes the turn: answers the pending tool_calls (otherwise the next
        message would fail in the API) and tells the user."""
        pending = [
            ToolMessage(content="No ejecutada: límite de pasos alcanzado.", tool_call_id=c["id"],
                        name=c["name"], status="error")
            for c in state["messages"][-1].tool_calls
        ]
        notice = AIMessage(content="He alcanzado el límite de pasos para esta pregunta. "
                                   "¿Puedes concretarla o dividirla en partes?")
        return {"messages": pending + [notice]}

    # ------------------------------------------------------------------ edges
    def after_model(state):
        if not state["messages"][-1].tool_calls:
            return END                                          # it has answered
        if llm_calls_this_turn(state["messages"]) >= max_llm_calls:
            return "limit"                                      # anti-loop brake
        return "approval"

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("approval", approval_node)   # decides with Command(goto=...): no outgoing edges
    graph.add_node("tools", tools_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["approval", "limit", END])
    graph.add_edge("tools", "model")
    graph.add_edge("limit", END)
    return graph.compile(checkpointer=checkpointer)
