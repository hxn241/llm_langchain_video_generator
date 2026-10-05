"""RAG: answer questions with the most similar chunks of an index.

    question -> search (k chunks) -> prompt with those chunks -> model -> answer

Pieces:
- create_retriever():           similarity or MMR (variety), with an optional metadata filter
- create_self_query():          the LLM extracts from the question WHAT to search and IN WHICH
                                document (like LangChain's SelfQueryRetriever, with structured output)
- create_search():              combines the two above
- create_rag():                 a single question -> answer
- create_conversational_rag():  question + history -> answer.
                                Before searching it rewrites the question so it stands
                                on its own ("¿y cuánto cuesta?" -> "¿cuánto cuesta la freidora?")
- rag_tool():                   a tool that an AGENT decides when to use. It returns the
                                chunks, not an answer: the agent writes the answer itself
                                (1 call fewer).

The default prompts are in Spanish. Every function accepts its own: instructions= (the RAG's rules:
language, tone, length, what to say when it does not know), rewrite_prompt=,
self_query_prompt=. filter= limits every search to some metadata (e.g. one source).
"""
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llmkit.documents import format_context, metadata_filter, metadata_values, source_summaries

I_DONT_KNOW = "No lo sé con la información disponible."

RAG_INSTRUCTIONS = (
    "Respondes preguntas usando SOLO el contexto que te dan. Cada trozo del contexto "
    "empieza con su fuente entre corchetes. Si la respuesta no está en el contexto, di "
    f"exactamente: \"{I_DONT_KNOW}\" Responde en español, de forma breve."
)

# The instructions go as a SystemMessage (literal text): braces in them are not
# taken as template variables.
def rag_prompt(instructions=RAG_INSTRUCTIONS):
    return ChatPromptTemplate.from_messages([
        SystemMessage(instructions),
        ("human", "Contexto:\n{context}\n\nPregunta: {question}"),
    ])


RAG_PROMPT = rag_prompt()


# ------------------------------------------------------------------ search
def create_retriever(index, k=4, search_type="similarity", filter=None):
    """search_type="similarity": the k most similar chunks (they may repeat the same thing).
    search_type="mmr": Maximum Marginal Relevance. Fetches fetch_k candidates and picks k
    that are relevant AND different from each other.
    filter: {"source": "manual_coffee_maker.md"} -> only searches those chunks."""
    options = {"k": k}
    if search_type == "mmr":
        options["fetch_k"] = max(20, 4 * k)
    elif search_type != "similarity":
        raise ValueError(f"Unknown search_type: {search_type!r} (use 'similarity' or 'mmr')")
    if filter:
        options["filter"] = metadata_filter(index, filter)
    return index.as_retriever(search_type=search_type, search_kwargs=options)


class SelfQuery(BaseModel):
    """What to search and, if the question says so, in which document."""
    query: str = Field(description="Lo que hay que buscar, sin la parte que dice de qué documento es")
    source: str = Field(default="", description="Nombre EXACTO de uno de los documentos de la lista si la "
                                                "pregunta se refiere a él; vacío si no se refiere a ninguno")


SELF_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Preparas búsquedas en una colección de documentos. Documentos disponibles:\n{sources}\n\n"
               "Cada documento aparece con su nombre de archivo y, detrás, su título. Si la pregunta "
               "se refiere a uno de ellos, pon en 'source' su nombre de archivo EXACTO (el que va "
               "antes de los dos puntos). Si no se refiere a ninguno, deja 'source' vacío."),
    ("human", "{question}"),
])


def create_self_query(model, sources, descriptions=None, prompt=SELF_QUERY_PROMPT):
    """question -> SelfQuery. One LLM call per question.
    descriptions: {source: short text} shown next to each name (e.g. its title).
    prompt: its own prompt, with the variables {sources} and {question}.
    If the model makes up a source that does not exist, it is ignored (no filter)."""
    chain = prompt | model.with_structured_output(SelfQuery)
    descriptions = descriptions or {}
    listing = "\n".join(f"- {s}: {descriptions[s]}" if descriptions.get(s) else f"- {s}" for s in sources)

    def self_query(question):
        r = chain.invoke({"question": question, "sources": listing})
        return SelfQuery(query=r.query or question, source=r.source if r.source in sources else "")

    return self_query


def create_search(index, k=4, search_type="similarity", self_query_model=None, verbose=False,
                  filter=None, self_query_prompt=SELF_QUERY_PROMPT):
    """Returns search(question) -> documents.
    With self_query_model, the LLM decides before searching whether to filter by source.
    filter: a fixed metadata filter applied to every search (e.g. {"source": "x.md"})."""
    if self_query_model is None:
        return create_retriever(index, k, search_type, filter).invoke

    self_query = create_self_query(self_query_model, metadata_values(index, "source"), source_summaries(index),
                                   prompt=self_query_prompt)

    def search(question):
        sq = self_query(question)
        fields = {**(filter or {}), **({"source": sq.source} if sq.source else {})}
        if verbose:
            print(f"  (self-query: search {sq.query!r}{f' only in {sq.source}' if sq.source else ''})")
        return create_retriever(index, k, search_type, fields or None).invoke(sq.query)

    return search


# ------------------------------------------------------------------ answer
def create_rag(index, model, k=4, search_type="similarity", self_query=False, verbose=False,
               instructions=RAG_INSTRUCTIONS, filter=None, self_query_prompt=SELF_QUERY_PROMPT):
    """Returns answer(question) -> (answer, documents).
    instructions: the system prompt (rules, language, tone, what to say if it does not know)."""
    search = create_search(index, k, search_type, model if self_query else None, verbose,
                           filter, self_query_prompt)
    chain = rag_prompt(instructions) | model | StrOutputParser()

    def answer(question):
        documents = search(question)                                            # 1) retrieve
        context = format_context(documents)                                     # 2) augment
        response = chain.invoke({"context": context, "question": question})     # 3) generate
        return response, documents

    return answer


# The history goes as TEXT inside one message, not as separate messages
# (MessagesPlaceholder). With separate messages, small reasoning models
# (e.g. nemotron-3-nano-4b in LM Studio) return an EMPTY answer.
# The examples come from "another store" on purpose: with products from ours,
# the small model copied the example's product instead of the conversation's.
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Reescribe la última pregunta del usuario para que se entienda sola, SIN la "
               "conversación. Si le falta el tema o la acción, cógelos de la conversación. Si ya se "
               "entiende sola, déjala igual. Devuelve SOLO la pregunta, en español.\n\n"
               "Ejemplos (de otra tienda):\n"
               "- Conversación sobre la mesa. Última pregunta: ¿y cuánto pesa? -> ¿Cuánto pesa la mesa?\n"
               "- Usuario preguntó: ¿Cómo se monta la mesa? Última pregunta: ¿Y la silla? -> "
               "¿Cómo se monta la silla?"),
    ("human", "Conversación:\n{conversation}\n\nÚltima pregunta: {question}"),
])

def rag_chat_prompt(instructions=RAG_INSTRUCTIONS):
    return ChatPromptTemplate.from_messages([
        SystemMessage(instructions),
        ("human", "Conversación hasta ahora:\n{conversation}\n\nContexto:\n{context}\n\nPregunta: {question}"),
    ])


RAG_CHAT_PROMPT = rag_chat_prompt()


def history_as_text(history, labels=("Usuario", "Asistente"), empty="(ninguna)"):
    """[HumanMessage, AIMessage, ...] -> 'Usuario: ...\\nAsistente: ...'"""
    user, assistant = labels
    return "\n".join(f"{user if m.type == 'human' else assistant}: {m.content}" for m in history) or empty


def create_conversational_rag(index, model, k=4, search_type="similarity", window=6,
                              instructions=RAG_INSTRUCTIONS, rewrite_prompt=REWRITE_PROMPT, filter=None):
    """Returns answer(question, history) -> (answer, documents, query).

    history: list of messages (HumanMessage/AIMessage); the last `window` are
    used. The caller keeps it (the memory is not inside the chain).
    query: the rewritten question used to search (the "generated_question"
    of ConversationalRetrievalChain).
    rewrite_prompt: its own prompt, with the variables {conversation} and {question}.

    Calls per turn: 2 (rewrite + answer); 1 on the first turn.
    """
    rewrite = rewrite_prompt | model | StrOutputParser()
    search = create_search(index, k, search_type, filter=filter)
    chain = rag_chat_prompt(instructions) | model | StrOutputParser()

    def answer(question, history=()):
        conversation = history_as_text(list(history)[-window:])
        query = question
        if history:
            query = rewrite.invoke({"conversation": conversation, "question": question}).strip() or question
        documents = search(query)
        # Answer the REWRITTEN question (like ConversationalRetrievalChain):
        # a bare "¿Y la cafetera?" confuses the model even with the conversation.
        response = chain.invoke({"conversation": conversation, "context": format_context(documents),
                                 "question": query})
        return response, documents, query

    return answer


def rag_tool(index, name, description, k=4, search_type="similarity", filter=None,
             nothing_found="No se encontró nada."):
    """Turns an index into a tool for agents. The description is what the model reads."""
    retriever = create_retriever(index, k, search_type, filter)

    @tool(name, description=description)
    def search_documents(question: str) -> str:
        return format_context(retriever.invoke(question)) or nothing_found

    return search_documents
