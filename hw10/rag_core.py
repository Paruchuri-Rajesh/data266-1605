"""Core pieces of the HW10 RAG pipeline (imported by the notebook).

Preprocessing, two chunking strategies, embedding + ChromaDB indexing,
retrieval with/without cross-encoder reranking, a simple conversational
memory, and Precision@K / Recall@K / MRR.
"""
import re

# ----------------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------------
# The arXiv articles are lower-cased, whitespace-tokenised LaTeX dumps
# ("long - distance", "refs.@xcite", "( @xmath0 )"). Clean them into
# normal prose so both the embedder and the sentence splitter behave.

def clean_text(t: str) -> str:
    t = re.sub(r"@xmath\d+", " ", t)              # inline maths placeholders
    t = re.sub(r"@xcite", " ", t)                 # citation placeholders
    t = re.sub(r"\brefs?\s*\.\s*(?=[\s,.\-]|$)", " ", t)  # dangling "refs ."
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+-\s+", "-", t)                # "long - distance" -> "long-distance"
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)     # no space before punctuation
    t = re.sub(r"([(\[])\s+", r"\1", t)           # no space after opening bracket
    for _ in range(3):                            # drop brackets emptied by the above
        t = re.sub(r"\(\s*[,;\-]*\s*\)|\[\s*[,;\-]*\s*\]", " ", t)
        t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)
    t = re.sub(r"([,;])(?:\s*[,;])+", r"\1", t)   # ", ," -> ","
    t = re.sub(r"\s*=\s*", " = ", t)
    return t.strip()


_ABBREV = re.compile(r"\b(?:e\.g|i\.e|et al|fig|figs|eq|eqs|ref|sec|vs|cf|no|approx|resp)\.$")


def split_sentences(t: str):
    """Regex sentence splitter that does not break after common abbreviations."""
    parts = re.split(r"(?<=[.!?])\s+", t)
    out, buf = [], ""
    for p in parts:
        buf = f"{buf} {p}".strip() if buf else p
        if _ABBREV.search(buf) or len(buf.split()) < 3:
            continue  # keep accumulating: abbreviation or tiny fragment
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


# ----------------------------------------------------------------------------
# 2. Chunking strategies
# ----------------------------------------------------------------------------

def chunk_fixed(text: str, size: int = 200, overlap: int = 40):
    """Strategy 1 — fixed-size sliding window over words.
    Every chunk is exactly `size` words (except the last) and consecutive
    chunks share `overlap` words. Ignores sentence boundaries."""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_sentences(text: str, max_words: int = 200, overlap_sents: int = 1):
    """Strategy 2 — sentence-aware packing.
    Whole sentences are packed greedily until adding the next one would exceed
    `max_words`; the last `overlap_sents` sentences are repeated at the start of
    the next chunk. A chunk never starts or ends mid-sentence, except that a
    run-on "sentence" longer than `max_words` (typically an equation or
    reference dump) is first split recursively into `max_words` word pieces."""
    sents = []
    for s in split_sentences(text):
        w = s.split()
        if len(w) <= max_words:
            sents.append(s)
        else:
            sents.extend(" ".join(w[j:j + max_words]) for j in range(0, len(w), max_words))
    chunks, cur = [], []
    i = 0
    while i < len(sents):
        s = sents[i]
        cur_len = sum(len(x.split()) for x in cur)
        if cur and cur_len + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            # if the overlap alone plus s still overflows, drop the overlap
            if sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
                cur = []
            continue
        cur.append(s)
        i += 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ----------------------------------------------------------------------------
# 3. Conversational memory
# ----------------------------------------------------------------------------

class ConversationMemory:
    """Keeps the last `max_turns` user queries (and what was retrieved for
    them). A follow-up query is made self-contained by prepending the
    remembered queries before it is embedded and reranked — a simple,
    LLM-free form of history-aware query rewriting."""

    def __init__(self, max_turns: int = 2):
        self.max_turns = max_turns
        self.turns = []  # list of {"query", "retrieved_ids"}

    def contextualize(self, query: str) -> str:
        history = [t["query"] for t in self.turns[-self.max_turns:]]
        return " ".join(history + [query])

    def add(self, query: str, retrieved_ids):
        self.turns.append({"query": query, "retrieved_ids": list(retrieved_ids)})

    def clear(self):
        self.turns = []


# ----------------------------------------------------------------------------
# 4. Metrics
# ----------------------------------------------------------------------------

def precision_at_k(ranked_rel, k):
    return sum(ranked_rel[:k]) / k


def recall_at_k(ranked_rel, k, n_relevant):
    return sum(ranked_rel[:k]) / n_relevant if n_relevant else 0.0


def reciprocal_rank(ranked_rel, k=None):
    lst = ranked_rel if k is None else ranked_rel[:k]
    for i, r in enumerate(lst, 1):
        if r:
            return 1.0 / i
    return 0.0
