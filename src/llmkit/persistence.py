"""Persistence: the two kinds of memory of LangGraph.

SHORT-TERM memory -> checkpointers: the state of each THREAD after every step.

    in_memory()          lost when the program closes (tests, quick trials)
    in_sqlite("name")    a file .state/name.sqlite: survives restarts.
                         A conversation, or a workflow paused waiting for
                         approval, can be resumed another day.
    (production with several servers: PostgresSaver, same interface)

Without a checkpointer there is no memory between calls, no interrupts and no history.

LONG-TERM memory -> stores: facts that are kept ACROSS threads (e.g. about a user).
Each item lives in a namespace (a tuple, e.g. ("memories", "ana")) under a key.
With embeddings, the store can search by MEANING ("¿qué bebo?" -> "me gusta el café").

    store_in_memory()           lost when the program closes
    store_in_sqlite("name")     a file .state/name.sqlite (semantic search with sqlite-vec)
    (production: PostgresStore with pgvector, same interface)
"""
import sqlite3

from langgraph.checkpoint.memory import InMemorySaver

from llmkit import config


def in_memory():
    return InMemorySaver()


def in_sqlite(name):
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = config.STATE_DIR / f"{name}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may use the connection from another thread
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def thread_config(thread):
    """The config that identifies a conversation or run (thread_id)."""
    return {"configurable": {"thread_id": str(thread)}}


# ------------------------------------------------------------ long-term memory
def _index(embeddings, fields):
    """The store's search index: which model turns text into vectors, and which
    fields of each item are embedded. embeddings=False -> no semantic search."""
    if embeddings is False:
        return None
    embeddings = embeddings or config.create_embeddings()
    dims = len(embeddings.embed_query("dimensions"))      # 1 call: each model has its own size (nomic: 768)
    return {"embed": embeddings, "dims": dims, "fields": list(fields)}


def store_in_memory(embeddings=None, fields=("text",)):
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore(index=_index(embeddings, fields))


def store_in_sqlite(name, embeddings=None, fields=("text",)):
    """.state/<name>.sqlite. Use a different name from your checkpointer's file."""
    from langgraph.store.sqlite import SqliteStore

    path = config.STATE_DIR / f"{name}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: autocommit, as SqliteStore expects
    store = SqliteStore(sqlite3.connect(path, check_same_thread=False, isolation_level=None),
                        index=_index(embeddings, fields))
    store.setup()                                          # creates its tables the first time
    return store
