"""Documents: load files, chunk them and store them in a vector index.

    files (.md .txt .csv .pdf .html; more with LOADERS) or a URL
        -> load_documents()    a list of Document (text + metadata)
        -> chunk_documents()   chunks of ~500 characters; Markdown split by section
        -> load_index()        one embedding per chunk, stored on disk

Document is LangChain's common format: page_content (the text) and metadata
(where it comes from: file, row, page, section...). The metadata travels with
every chunk: RAG uses it to cite the source and to FILTER the search.

The index is stored on disk in Chroma (VECTOR_STORE=chroma, the default) or kept
only in RAM while the program runs (VECTOR_STORE=memory: tests and quick trials;
it is rebuilt the next time). Every chunk has a STABLE id computed from its text and metadata:
when re-indexing, only the embeddings of new chunks are computed and chunks
that no longer exist are deleted. Unchanged chunks cost nothing.
"""
import csv
import hashlib
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from llmkit import config

# -------------------------------------------------------------------- load
# A loader is a function (path, source) -> [Document]. `source` is the name that
# goes into the metadata (the file name, or "subfolder/file" inside a folder).
def _read_text(path):
    """UTF-8, or the Windows encoding if the file was saved with it (accents)."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252")


def _load_text(path, source):
    return [Document(page_content=_read_text(path), metadata={"source": source})]


def _load_csv(path, source):
    """One Document per row, formatted as 'column: value' (what CSVLoader does)."""
    rows = csv.DictReader(io.StringIO(_read_text(path), newline=""))   # keeps multi-line cells
    return [
        Document(page_content="\n".join(f"{k}: {v}" for k, v in row.items()),
                 metadata={"source": source, "row": i})
        for i, row in enumerate(rows, start=1)
    ]


def _load_pdf(path, source):
    """One Document per page (like PyPDFLoader). Needs: python -m pip install pypdf"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(f"To read {path.name} install:  python -m pip install pypdf") from e
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": source, "page": n})
        for n, page in enumerate(PdfReader(path).pages, start=1)
    ]


def _load_html(path, source):
    return [Document(page_content=html_to_text(_read_text(path)), metadata={"source": source})]


# Extension -> loader. To read another format, add yours (before loading):
#     documents.LOADERS[".docx"] = my_docx_loader      # (path, source) -> [Document]
LOADERS = {
    ".md": _load_text, ".markdown": _load_text, ".txt": _load_text,
    ".csv": _load_csv, ".pdf": _load_pdf, ".html": _load_html, ".htm": _load_html,
}


def load_documents(path, recursive=True):
    """Loads one file, or every supported file in a folder (and its subfolders).

    Files in subfolders get "subfolder/file.md" as their source. Hidden files
    and folders (starting with ".") are skipped.
    """
    path = Path(path)
    if path.is_dir():
        candidates = path.rglob("*") if recursive else path.iterdir()
        files = sorted(
            p for p in candidates
            if p.is_file() and p.suffix.lower() in LOADERS
            and not any(part.startswith(".") for part in p.relative_to(path).parts)
        )
        sources = [p.relative_to(path).as_posix() for p in files]
    else:
        files, sources = [path], [path.name]
    documents = []
    for file, source in zip(files, sources):
        loader = LOADERS.get(file.suffix.lower(), _load_text)
        documents.extend(loader(file, source))
    return documents


class _TextOnly(HTMLParser):
    """Keeps the visible text of an HTML page (no scripts or styles)."""
    IGNORE = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self.parts, self._inside = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORE:
            self._inside.append(tag)

    def handle_endtag(self, tag):
        if self._inside and self._inside[-1] == tag:
            self._inside.pop()

    def handle_data(self, data):
        if not self._inside and data.strip():
            self.parts.append(data.strip())


def html_to_text(html):
    """The visible text of an HTML page, one piece per line."""
    reader = _TextOnly()
    reader.feed(html)
    return "\n".join(reader.parts)


def load_web(url, timeout=20):
    """A web page -> one Document with its text (what WebBaseLoader does, without
    BeautifulSoup). The text comes out "dirty" (menus, footers...): that is
    normal when loading web pages, which is why you should check it before indexing."""
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": config.USER_AGENT})
    r.raise_for_status()
    return [Document(page_content=html_to_text(r.text), metadata={"source": url})]


# ------------------------------------------------------------------- chunk
MARKDOWN_EXTENSIONS = (".md", ".markdown")


def chunk_documents(documents, chunk_size=500, chunk_overlap=50, markdown_headers=(("##", "section"),)):
    """Splits documents into chunks:

    - CSV rows: each one is already a meaningful chunk (a product).
    - Markdown: first by section and then by size. The section goes into the
      metadata, so every chunk knows where it comes from.
      markdown_headers: which headings split and the metadata key of each one.
      E.g. (("#", "chapter"), ("##", "section")).
    - Everything else (txt, pdf, html): by size.
    """
    by_size = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    by_section = MarkdownHeaderTextSplitter(list(markdown_headers), strip_headers=False)
    chunks = []
    for doc in documents:
        if "row" in doc.metadata:
            chunks.append(doc)
        elif doc.metadata.get("source", "").lower().endswith(MARKDOWN_EXTENSIONS):
            sections = by_section.split_text(doc.page_content)
            for s in sections:
                s.metadata = {**doc.metadata, **s.metadata}      # source + section
            chunks.extend(by_size.split_documents(sections))
        else:
            chunks.extend(by_size.split_documents([doc]))
    return chunks


def citation(doc):
    """'manual_coffee_maker.md, Limpieza' / 'catalog.csv, row 3' / 'x.pdf, page 2'"""
    m = doc.metadata
    parts = [str(m.get("source", "?"))]
    if "row" in m:
        parts.append(f"row {m['row']}")
    if "page" in m:
        parts.append(f"page {m['page']}")
    if "section" in m:
        parts.append(m["section"])
    return ", ".join(parts)


def format_context(documents):
    """Chunks -> text for the prompt, each one starting with its source in brackets."""
    return "\n\n".join(f"[{citation(doc)}]\n{doc.page_content}" for doc in documents)


# ------------------------------------------------------------ vector index
def chunk_id(chunk):
    """Stable id: the same text with the same metadata always gives the same id.
    That way we know which chunks are already indexed without recomputing
    anything, and a repeated chunk (e.g. a file loaded twice) is stored once."""
    content = json.dumps([chunk.page_content, chunk.metadata], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def index_name(name, embedding_model=None):
    """The name includes the embedding model: vectors from different models
    cannot be mixed, so each model gets its own index."""
    model = (embedding_model or config.embedding_model_name()).split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", f"{name}__{model}")


# VECTOR_STORE=memory: the indexes of this run, so a second load_index of the same
# index in the same program reuses it (and only recomputes what changed).
# Nothing is written to disk: for that, use Chroma.
_MEMORY_INDEXES = {}


def open_index(name, embeddings=None, store=None):
    """Opens the index `name` (empty if it does not exist). Computes no embeddings."""
    embeddings = embeddings or config.create_embeddings()
    store = store or config.VECTOR_STORE

    if store == "chroma":
        try:
            from chromadb.config import Settings
            from langchain_chroma import Chroma
        except ImportError as e:
            raise RuntimeError("To use Chroma install:  python -m pip install langchain-chroma  "
                               "(or set VECTOR_STORE=memory in the .env)") from e
        return Chroma(
            collection_name=index_name(name),
            embedding_function=embeddings,
            persist_directory=str(config.STATE_DIR / "chroma"),
            collection_metadata={"hnsw:space": "cosine"},     # cosine similarity (default: L2 distance)
            client_settings=Settings(anonymized_telemetry=False),
        )

    if store == "memory":
        key = (str(config.STATE_DIR), index_name(name))         # one per project state and model
        if key not in _MEMORY_INDEXES:
            _MEMORY_INDEXES[key] = InMemoryVectorStore(embeddings)
        return _MEMORY_INDEXES[key]

    raise ValueError(f"Unknown VECTOR_STORE: {store!r} (use 'chroma' or 'memory')")


def stored_ids(index):
    if isinstance(index, InMemoryVectorStore):
        return set(index.store)
    return set(index.get(include=[])["ids"])


def stored_metadatas(index):
    if isinstance(index, InMemoryVectorStore):
        return [d["metadata"] for d in index.store.values()]
    return index.get(include=["metadatas"])["metadatas"]


def metadata_values(index, field):
    """Distinct values of a metadata field in the index. E.g. every source."""
    return sorted({m[field] for m in stored_metadatas(index) if field in m})


def source_summaries(index, length=80):
    """{source: first line of its first chunk}. Helps an LLM tell the documents
    apart when the file names are not enough (e.g. English names, Spanish questions)."""
    if isinstance(index, InMemoryVectorStore):
        pairs = [(d["metadata"], d["text"]) for d in index.store.values()]
    else:
        stored = index.get(include=["metadatas", "documents"])
        pairs = zip(stored["metadatas"], stored["documents"])
    first_chunks = {}
    for metadata, text in sorted(pairs, key=lambda p: (p[0].get("row", 0), p[0].get("page", 0))):
        first_chunks.setdefault(metadata.get("source"), text)
    return {source: next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")[:length]
            for source, text in sorted(first_chunks.items()) if source}


def metadata_filter(index, fields):
    """{"source": "x.md"} -> the filter in each store's format:
    Chroma uses a dict ($and for several fields); InMemoryVectorStore, a function."""
    if isinstance(index, InMemoryVectorStore):
        return lambda doc: all(doc.metadata.get(k) == v for k, v in fields.items())
    conditions = [{k: v} for k, v in fields.items()]
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class IndexNotAvailable(RuntimeError):
    """There is no index for the current documents and it must not be created here."""


def load_index(folder, name, embeddings=None, verbose=True, create_if_missing=True,
               how_to_index="call load_index() with create_if_missing=True", **chunk_options):
    """Returns the index of `folder`, up to date with its documents.

    chunk_options: passed to chunk_documents (chunk_size=, chunk_overlap=,
    markdown_headers=). Changing them changes the chunks, so they get re-indexed.

    Compares the ids of the current chunks with the stored ones:
      - new or modified chunks -> computes ONLY their embeddings and adds them
      - chunks that no longer exist -> deletes them
      - the rest -> cost nothing

    create_if_missing=False separates the phases: a QUERY only loads, and if the
    index is missing or the documents changed, it says so instead of indexing
    on its own (a user's question never triggers an indexing run).
    Loading and chunking the documents to compute the ids makes no API calls.
    how_to_index: what to tell the user in that case (e.g. your program's command).
    """
    chunks = {chunk_id(c): c for c in chunk_documents(load_documents(folder), **chunk_options)}
    index = open_index(name, embeddings)
    stored = stored_ids(index)
    new = [i for i in chunks if i not in stored]
    stale = sorted(stored - chunks.keys())

    if not new and not stale:
        if verbose:
            print(f"Index '{name}' up to date ({len(chunks)} chunks, nothing recomputed)")
        return index

    if not create_if_missing:
        reason = "the documents changed since the last indexing" if stored else "it has not been indexed yet"
        raise IndexNotAvailable(f"No index for '{name}': {reason}. To index: {how_to_index}")

    if stale:
        index.delete(ids=stale)
    if new:
        if verbose:
            print(f"Computing embeddings for {len(new)} chunks...")
        index.add_documents([chunks[i] for i in new], ids=new)
    if verbose:
        print(f"Index '{name}': {len(new)} new chunks, {len(stale)} deleted, "
              f"{len(chunks) - len(new)} unchanged")
    return index
