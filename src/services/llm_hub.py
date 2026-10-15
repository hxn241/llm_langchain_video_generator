"""LLMHub: one object for everything you build with an LLM.

It does not add new logic: it gathers what is spread across llmkit (model,
documents, RAG, tools, agent, memory, approvals, evaluation) behind one class,
so a new project only needs one import. Nothing in it is tied to one product:
every prompt, description and option can be passed in.

    from services import LLMHub

    hub = LLMHub()                                # model from config.yaml (llm.default), memory in RAM
    hub = LLMHub(memory="checkpoints")            # memory in <STATE_DIR>/checkpoints.sqlite (survives restarts)
    hub = LLMHub(temperature=0.8, max_tokens=2000)   # any option of the default model

    # Model per node (config.yaml: llm.<node> inherits from llm.default)
    hub.extract(article, Script, prompt, node="write_script")
    hub.unload_local_models()                     # lms unload --all: frees the VRAM

    # 1. Direct question
    hub.ask("¿Qué es un modelo de difusión?")

    # 2. Chain: text in, text or data out
    summarize = hub.chain("Resume la noticia en una frase.")
    summarize.invoke({"text": "..."})             # -> "..."
    hub.extract(article, Script, "Escribe el guion del vídeo.")   # -> Script(title=..., scenes=[...])
    hub.look("Eliges imágenes.", "¿Cuál encaja?", [jpeg_bytes], schema=Choice, node="visuals")  # vision

    # 3. Your documents (any folder; subfolders and .md .txt .csv .pdf .html)
    answer = hub.rag("data/docs")                 # answer("question") -> (response, sources)
    answer = hub.rag("docs", instructions="Answer in English, in detail. ...",
                     chunking={"chunk_size": 1000})
    chat = hub.rag("data/docs", conversational=True)

    # 4. Agent (the model chooses the tools)
    agent = hub.agent([calculator, hub.web_tool(),
                       hub.docs_tool("data/docs", "search_docs", "Busca en ...")],
                      "Eres el investigador de ...", approval={"upload_video"})
    hub.chat(agent, "¿Qué ha anunciado OpenAI hoy?", thread="research-42")

    # 5. Workflow (your own StateGraph)
    def review(state):                            # a node that asks a human
        decision = hub.ask_approval("publish", {"title": state["title"]})
        return {"approved": decision["type"] == "approve"}
    workflow = hub.workflow(graph)                # compiles it with the hub's memory
    hub.run(workflow, {"format": "short"}, thread="run-1")   # -> final state

    # 6. Long-term memory (facts kept ACROSS threads, per user)
    hub = LLMHub(memory="chat", long_term="facts")   # chat.sqlite + facts.sqlite
    hub.remember("canal", "Los Shorts sobre GPUs funcionan mejor")
    hub.recall("canal", "¿qué temas funcionan?")  # -> [{"key", "text", "kind", "score", ...}] by meaning
    hub.memories("canal") / hub.forget("canal", key)

    # Cross-cutting: works with any agent or workflow
    hub.chat / hub.run          threads (memory) + pauses for human approval
    hub.evaluate(...)           evaluation with judges and a report
"""
import datetime
import hashlib
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from llmkit import approval, config, persistence
from llmkit.approval import approve_all, reject_all
from llmkit.console import ask_decisions, describe_error, pending_interrupts, run_cli, run_with_approvals
from llmkit.documents import load_index
from llmkit.rag import create_conversational_rag, create_rag, rag_tool
from llmkit.web_search import DEFAULT_DESCRIPTION, create_search_tool, search
from services import costs, llm


def _data_url(image):
    """bytes or a path (JPEG/PNG) -> data:image/...;base64,... (how images travel in the OpenAI API)."""
    import base64

    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    kind = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
    return f"data:image/{kind};base64,{base64.b64encode(data).decode()}"


class LLMHub:
    def __init__(self, model=None, temperature=None, memory=None, long_term=None, embeddings=None,
                 **model_options):
        """model: a chat model. Default: llm.default of config.yaml if it exists;
               otherwise MODEL_NAME from the .env.
        temperature, model_options: override those of the default model (max_tokens=, timeout=...).
        memory: SHORT-term memory (threads). None = in RAM (lost on exit); "name" = <STATE_DIR>/name.sqlite.
        long_term: LONG-term memory (facts across threads). None = not used; True = in RAM;
                   "name" = <STATE_DIR>/name.sqlite. Use a different name from `memory`.
        embeddings: for the long-term memory's search by meaning (default: the .env's model).
        """
        if temperature is not None:
            model_options["temperature"] = temperature
        self._node_models = {}
        self._failed_nodes = set()      # nodes whose model failed in this run: they use their fallback
        self._from_config = model is None and llm.CONFIG_FILE.exists()   # models of config.yaml
        if model is not None:
            self.model = model
        elif self._from_config:
            self.model = llm.model_for("default", **model_options)
        else:
            self.model = config.create_model(**{"temperature": 0, **model_options})
        self.checkpointer = persistence.in_sqlite(memory) if memory else persistence.in_memory()
        self._long_term = long_term
        self._embeddings = embeddings
        self._store = None
        self._indexes = {}

    # ------------------------------------------------------- model per node
    def model_for(self, node=None):
        """The model of a graph node (config.yaml: llm.<node>, inheriting from llm.default).
        node=None -> this hub's default model. Each node's model is created once."""
        if node is None:
            return self.model
        if node not in self._node_models:
            self._node_models[node] = llm.model_for(node, callbacks=[costs.UsageLog(node)])
        return self._node_models[node]

    def structured(self, schema, node=None):
        """The model of `node` answering with an object of `schema`, asked as config.yaml says
        (llm.<node>.structured_output: json_schema or function_calling).

        llm.<node>.fallback: another node (e.g. a local model) used when this one fails (a free
        OpenRouter model saturated: 429, its daily limit...). After the first failure the rest of
        this run goes straight to the fallback (no more waits on a saturated model)."""
        model = self.model_for(node)
        if not (self._from_config or node is not None):
            return model.with_structured_output(schema)
        primary = model.with_structured_output(schema, method=llm.structured_method(node or "default"))
        backup_node = llm.node_settings(node or "default").get("fallback")
        if not backup_node:
            return primary
        backup = self.structured(schema, backup_node)

        def call(inputs):
            if node not in self._failed_nodes:
                try:
                    return primary.invoke(inputs)
                except Exception as e:                       # anything: the run goes on with the fallback
                    print(f"  · {node}: {model.model_name} falló ({describe_error(e)[:160]}); "
                          f"sigo con {backup_node} ({llm.node_settings(backup_node)['model']})")
                    self._failed_nodes.add(node)
            self.prepare(backup_node)                        # e.g. loads the local model in LM Studio
            return backup.invoke(inputs)

        return RunnableLambda(call)

    def prepare(self, node=None):
        """Right before calling the model of `node`: loads it in LM Studio as config.yaml says
        (context...). ask/chain/agent call it on their own."""
        if self._from_config or node is not None:
            llm.prepare(node or "default")

    @staticmethod
    def unload_local_models(verbose=True):
        """lms unload --all: frees the VRAM (e.g. before Flux/Chatterbox).
        False if LM Studio was not running. The next local call reloads the model."""
        return llm.unload_local_models(verbose=verbose)

    # ------------------------------------------------------------ 1. direct
    def ask(self, question, system=None, node=None):
        """One question, one answer (text). node: use that node's model (config.yaml)."""
        messages = [("system", system)] if system else []
        self.prepare(node)
        return self.model_for(node).invoke(messages + [("human", question)]).text

    # ------------------------------------------------------------- 2. chain
    def chain(self, system, human="{text}", schema=None, system_is_template=False, node=None):
        """prompt | model | output. Returns a runnable: .invoke({...}) or .batch([...]).

        schema=None -> text; schema=a Pydantic class -> an object with those fields.
        system: literal text, so braces are safe (e.g. a JSON example in a prompt .md).
                The {variables} go in `human`. system_is_template=True -> the system
                prompt also takes {variables} (and literal braces must be written {{ }}).
        node: use that node's model (config.yaml), e.g. node="write_script".
        """
        model = self.model_for(node)
        system_message = ("system", system) if system_is_template else self._system(system, node)
        prompt = ChatPromptTemplate.from_messages([system_message, ("human", human)])
        # Loads the model as config.yaml says on EVERY call (not now: building a graph must not load models)
        prepare = RunnableLambda(lambda inputs: (self.prepare(node), inputs)[1])
        if schema:
            return prepare | prompt | self.structured(schema, node)
        return prepare | prompt | model | StrOutputParser()

    def look(self, system, text, images, schema=None, node=None):
        """Text + images -> text, or an object of `schema`. Needs a model with vision
        (in LM Studio, type "vlm"). images: JPEG/PNG as bytes or file paths."""
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images]
        self.prepare(node)
        if schema:
            return self.structured(schema, node).invoke([self._system(system, node), HumanMessage(content)])
        return self.model_for(node).invoke([self._system(system, node), HumanMessage(content)]).text

    def _system(self, text, node):
        """The system prompt; for an Anthropic model, marked to be cached (llm.caches_prompts): the
        same instructions sent again within 5 minutes cost 10 %."""
        if node is not None and self._from_config and llm.caches_prompts(node):
            return SystemMessage([{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}])
        return SystemMessage(text)

    def extract(self, text, schema, instructions="Extrae los datos del texto.", node=None):
        """Text -> the fields of `schema` (classify, extract data...)."""
        return self.chain(instructions, schema=schema, node=node).invoke({"text": text})

    def batch(self, function, inputs, max_concurrency=2):
        """Runs function(input) for many inputs, `max_concurrency` at a time.

        If one fails, the rest continue: its place in the result list holds the
        exception instead (check with isinstance(r, Exception)).
        With LM Studio use max_concurrency=1.
        """
        return RunnableLambda(function).batch(list(inputs), config={"max_concurrency": max_concurrency},
                                              return_exceptions=True)

    # --------------------------------------------------------- 3. documents
    def index(self, folder, name=None, create_if_missing=True, how_to_index=None, **chunking):
        """Loads (or updates) the index of a folder of documents. Cached per folder.

        name: the index name (default: the folder's name + a short hash of its path,
              so two folders called the same do not share an index).
        create_if_missing=False: only loads. If the index is missing or the documents
              changed, it raises IndexNotAvailable (with how_to_index in the message)
              instead of computing embeddings: a question never triggers an indexing run.
        chunking: chunk_size=, chunk_overlap=, markdown_headers= (see documents.chunk_documents).
        """
        folder = Path(folder)
        if not folder.is_absolute():
            folder = config.BASE / folder
        folder = folder.resolve()
        name = name or f"{folder.name}_{hashlib.sha256(str(folder).encode()).hexdigest()[:6]}"
        key = (folder, name, repr(sorted(chunking.items())))
        if key not in self._indexes:
            self._indexes[key] = load_index(
                folder, name, create_if_missing=create_if_missing,
                how_to_index=how_to_index or "call hub.index() with create_if_missing=True", **chunking)
        return self._indexes[key]

    def _as_index(self, source, name, chunking):
        """A folder -> its index; an index already loaded -> itself."""
        if isinstance(source, (str, Path)):
            return self.index(source, name, **(chunking or {}))
        return source

    def rag(self, source, conversational=False, name=None, chunking=None, **options):
        """Questions about your documents.

        source: a folder, or an index you already loaded with hub.index(...).
        conversational=False -> answer(question) -> (response, sources)
        conversational=True  -> answer(question, history) -> (response, sources, query)
        options: instructions= (rules, language, tone), k=4, search_type="mmr",
                 filter={"source": ...}, self_query=True and verbose=True (only the simple one)...
        """
        index = self._as_index(source, name, chunking)
        if conversational:
            return create_conversational_rag(index, self.model, **options)
        return create_rag(index, self.model, **options)

    # ------------------------------------------------------------- tools
    def docs_tool(self, source, name, description, index_name=None, chunking=None, **options):
        """Your documents as a tool for an agent. The description is what the model reads.
        source: a folder or an index. options: k=, search_type=, filter=."""
        return rag_tool(self._as_index(source, index_name, chunking), name, description, **options)

    @staticmethod
    def web_tool(name="search_web", description=DEFAULT_DESCRIPTION, max_results=3, region=None):
        """A web search tool. Tell the model in `description` WHEN to search the web."""
        return create_search_tool(name, description, max_results, region)

    @staticmethod
    def web_search(query, max_results=3):
        """Searches the web from your own code (not as a tool).
        -> {"answer": str | None, "results": [{"url", "title", "content"}, ...]}"""
        return search(query, max_results)

    # ------------------------------------------------------------- 4. agent
    def agent(self, tools, system_prompt, approval=(), max_calls=config.MAX_LLM_CALLS, middleware=(),
              node=None):
        """An agent with create_agent (LangChain's ready-made one).

        approval: names of tools that need a human "yes" before running.
        max_calls: anti-loop brake (model calls per turn).
        middleware: any other create_agent middleware you want to add.
        node: use that node's model (config.yaml), e.g. node="research".
        """
        from langchain.agents import create_agent
        from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware

        self.prepare(node)     # agents: loaded once, when created
        stack = [ModelCallLimitMiddleware(run_limit=max_calls, exit_behavior="end"), *middleware]
        if approval:
            stack.insert(0, HumanInTheLoopMiddleware(interrupt_on=self._interrupt_on(approval)))
        return create_agent(self.model_for(node), tools, system_prompt=system_prompt,
                            checkpointer=self.checkpointer, middleware=stack,
                            store=self.store if self._long_term else None)

    _interrupt_on = staticmethod(approval.interrupt_on)

    # ---------------------------------------------------------- 5. workflow
    def workflow(self, graph):
        """Compiles your StateGraph with the hub's memory (needed for threads and interrupts)."""
        return graph.compile(checkpointer=self.checkpointer)

    # In a workflow node: pauses until a human decides; returns
    # {"type": "approve"} | {"type": "edit", "edited_action": {...}} | {"type": "reject", "message": ...}
    ask_approval = staticmethod(approval.ask_approval)

    # ---------------------------------------------- cross-cutting: run any graph
    def run(self, graph, input, thread="default", approve=ask_decisions, verbose=False):
        """Runs an agent or workflow in a thread and returns its final state.

        Every time it pauses for approval it calls approve(request): ask_decisions
        asks in the terminal; approve_all / reject_all decide on their own.
        input=None resumes a thread that was left paused.
        """
        thread_config = persistence.thread_config(thread)
        run_with_approvals(graph, input, thread_config, approve, verbose)
        return graph.get_state(thread_config).values

    def chat(self, graph, message, thread="default", approve=ask_decisions, verbose=False):
        """Sends a message to an agent and returns its reply (text). Same thread = it remembers."""
        state = self.run(graph, {"messages": [HumanMessage(message)]}, thread, approve, verbose)
        return state["messages"][-1].text

    # ------------------------------------------------------ 6. long-term memory
    @property
    def store(self):
        """The long-term memory store. Created the first time it is used (creating
        its search index costs one call to the embeddings model)."""
        if self._store is None:
            if not self._long_term:
                raise RuntimeError("This LLMHub has no long-term memory: create it with LLMHub(long_term=...)")
            if self._long_term is True:
                self._store = persistence.store_in_memory(self._embeddings)
            else:
                self._store = persistence.store_in_sqlite(self._long_term, self._embeddings)
        return self._store

    @staticmethod
    def _namespace(user):
        return ("memories", str(user))

    def remember(self, user, text, kind="fact"):
        """Saves a memory about `user` and returns its key.
        kind: what type of memory it is (e.g. "fact", "episode", "instruction")."""
        key = uuid.uuid4().hex[:8]
        self.store.put(self._namespace(user), key, {
            "text": text, "kind": kind, "created_at": datetime.datetime.now().isoformat(timespec="seconds")})
        return key

    def recall(self, user, query, limit=5, kind=None):
        """The memories of `user` most related to `query` (by meaning), best first."""
        items = self.store.search(self._namespace(user), query=query, limit=limit,
                                  filter={"kind": kind} if kind else None)
        return [{"key": i.key, **i.value, "score": i.score} for i in items]

    def memories(self, user, kind=None, limit=100):
        """All the memories of `user` (no search)."""
        items = self.store.search(self._namespace(user), limit=limit, filter={"kind": kind} if kind else None)
        return [{"key": i.key, **i.value} for i in items]

    def forget(self, user, key):
        self.store.delete(self._namespace(user), key)

    # ------------------------------------------- cross-cutting: look inside a thread
    @staticmethod
    def state(graph, thread="default"):
        """The current values of a thread ({} if it does not exist)."""
        return graph.get_state(persistence.thread_config(thread)).values

    def history(self, graph, thread="default"):
        """The messages of a thread (what the agent remembers)."""
        return self.state(graph, thread).get("messages", [])

    def tools_used(self, graph, thread="default"):
        """Names of the tools the agent asked for since the user's last message."""
        used = []
        for m in reversed(self.history(graph, thread)):
            if m.type == "human":
                break
            used += [c["name"] for c in getattr(m, "tool_calls", [])]
        return used

    @staticmethod
    def pending(graph, thread="default"):
        """Approval requests a thread is paused on ([] if none)."""
        return pending_interrupts(graph, persistence.thread_config(thread))

    @staticmethod
    def update(graph, thread, values, as_node=None):
        """Changes a thread's state by hand, e.g. a human's critique. With as_node the
        graph continues as if that node had written it (then hub.run(graph, None, thread))."""
        graph.update_state(persistence.thread_config(thread), values, as_node=as_node)

    # ---------------------------------------------------- cross-cutting: evaluate
    def evaluate(self, cases, system, criteria=None, name="llm_hub", judge_model=None, checks=(),
                 whole_case=False):
        """Evaluates `system` on some cases and prints a report.

        cases: a list of dicts or a file name in data/ (e.g. "eval_rag.json").
               Keys: "input", and "expected_answer" and/or "criterion" (the judges
               skip the cases without them).
        system: a function input_text -> answer_text, or -> a dict with "answer"
                and anything else your checks need.
        whole_case=True: system receives the whole case (e.g. to read its "history").
        criteria: list of rubric criteria (scored 1-5) to apply to every case.
        checks: your own checks, functions (case, result) -> (ok, reason).
                Deterministic ones (no LLM) are cheap and exact: use them when you can.
        """
        from llmkit import evaluation as ev

        if isinstance(cases, str):
            cases = ev.load_cases(cases)
        if judge_model is None:   # config.yaml: llm.judge (inherits from llm.default); without it, the .env
            judge_model = llm.model_for("judge") if llm.CONFIG_FILE.exists() else config.create_judge_model()
        judge = judge_model
        all_checks = [*checks, ev.check_with_reference(ev.reference_judge(judge)),
                      ev.check_with_criterion(ev.criterion_judge(judge))]
        if criteria:
            all_checks.append(ev.check_with_rubric(ev.rubric_judge(judge), criteria))

        def run_case(case):
            result = system(case if whole_case else case["input"])
            return result if isinstance(result, dict) else {"answer": result}

        rows = ev.evaluate(cases, run_case, all_checks)
        ev.report(rows, name)
        return rows


__all__ = ["LLMHub", "approve_all", "ask_decisions", "describe_error", "reject_all", "run_cli"]
