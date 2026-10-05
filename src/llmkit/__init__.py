"""llmkit: shared building blocks for LLM applications (used here by the video pipeline).

    config       .env variables, paths, create_model(), create_embeddings()
    documents    load files (any format via LOADERS, subfolders) or web pages, chunk
                 them (Markdown by section), Chroma or in-memory index that only
                 recomputes what changed, metadata filters
    rag          retriever (similarity/MMR + filter), self-query, single-question
                 and conversational RAG; RAG as a tool for agents
    web_search   web search (DuckDuckGo or Tavily) and search tools
    tools        general tools: calculator, wikipedia, today's date
    approval     the human-approval format shared by agents and workflows
    agent        agent graph with human approval
    persistence  checkpointers (memory or SQLite) and long-term stores
    storage      your own data (rows) in SQLite
    evaluation   LLM judges, test cases, report
    console      run graphs showing each step, ask for approval, API errors

Nothing here belongs to one application: its own tools, prompts and data
live in the application (e.g. src/graph/, src/prompts/). Every prompt has a Spanish default and can be
replaced (instructions=, prompt=, description=...).

Convention: identifiers, comments and docs are in English. Text the LLM reads
(prompts, tool docstrings, field descriptions) is in Spanish by default.
"""
