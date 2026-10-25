"""
Stages 3 and 4 of the pipeline: embedding chunks and retrieving them.

Three things in here are worth knowing about, because they'd quietly break the
rest of the project if they were wrong:

1. The Chroma collection is created with cosine distance, explicitly. Chroma
   defaults to squared L2, and the 0.6 threshold the course uses is calibrated
   against cosine. Getting this wrong makes every distance number meaningless.

2. `search` returns the distance alongside each chunk. Milestone 4 has you
   compare distances, so they have to be visible.

3. The embedding model is the one Chroma bundles, not one loaded through
   `sentence-transformers`. It is the same model — `all-MiniLM-L6-v2`, 384
   dimensions — but it arrives as an ONNX build from Chroma's own CDN, so the
   install needs neither PyTorch nor a reachable Hugging Face. See `_embedder`.
"""

import math
import os
import re
import shutil
from dataclasses import dataclass

# Must be set BEFORE chromadb is imported. Without it, some Chroma versions
# print "Failed to send telemetry event ..." on every single call — which looks
# exactly like a real error, isn't one, and cost a previous cohort a lot of
# confused help-channel messages.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402

import config
from chunker import Chunk


@dataclass
class Result:
    """One retrieved chunk and how far it was from the question."""

    text: str
    source: str
    label: str
    distance: float   # LOWER IS BETTER. 0.3 is close, 0.9 is unrelated.
    produced_by: str
    # Only filled in by hybrid search. HIGHER IS BETTER for both.
    bm25_score: float | None = None   # raw keyword score; 0 means no shared words
    fused_score: float | None = None  # Reciprocal Rank Fusion score; sets the order


_model = None

# The model Chroma bundles. Anything else in config.EMBEDDING_MODEL means
# "fetch that one from Hugging Face instead" — see `_embedder`.
BUNDLED_MODEL = "all-MiniLM-L6-v2"


class _OnnxEmbedder:
    """
    Chroma's built-in embedder, wrapped to look like the other two.

    Chroma's embedding functions are called directly and hand back numpy
    arrays. The rest of this file wants `.encode(texts)`, so the adapter lives
    here rather than making every caller care which embedder it got.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts, show_progress_bar: bool = False):
        return [vector.tolist() for vector in self._ef(list(texts))]


def _sentence_transformer(name: str):
    """
    The escape hatch: any model that isn't the bundled one.

    Unit 2's "try a second embedding model" stretch option comes through here,
    and so does anything you set `EMBEDDING_MODEL` to. This path *does* need
    `sentence-transformers` and a reachable Hugging Face, neither of which the
    default install has — which is the whole point of the default install.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"config.EMBEDDING_MODEL is set to {name!r}, which isn't the model "
            f"Chroma bundles ({BUNDLED_MODEL!r}), so it has to be downloaded "
            f"from Hugging Face.\n"
            f"Install the optional dependency first:\n"
            f"    pip install 'sentence-transformers>=3.4,<3.5'\n"
            f"Or set EMBEDDING_MODEL back to {BUNDLED_MODEL!r}."
        ) from exc

    return SentenceTransformer(name)


def _embedder():
    """
    Load the embedding model once and keep it.

    First call is slow — it downloads about 80 MB. That's why setup happens
    before class.
    """
    global _model

    if _model is not None:
        return _model

    # Used only by this repo's own smoke test, which runs where no model can be
    # downloaded at all. Never set this yourself.
    if os.getenv("AI201_FAKE_EMBEDDINGS") == "1":
        from _smoke_embedder import FakeEmbedder

        _model = FakeEmbedder()
    elif config.EMBEDDING_MODEL == BUNDLED_MODEL:
        _model = _OnnxEmbedder()
    else:
        _model = _sentence_transformer(config.EMBEDDING_MODEL)

    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Turn text into vectors. Runs on your machine, costs no API quota."""
    vectors = _embedder().encode(texts, show_progress_bar=False)
    # sentence-transformers and the smoke stand-in return something with a
    # .tolist(); _OnnxEmbedder has already done that conversion itself.
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def _client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def build_index(
    chunks: list[Chunk],
    corpus: str | None = None,
    variant: str = "default",
) -> int:
    """
    Embed every chunk and store it.

    `variant` lets you keep more than one index of the same corpus at the same
    time. In unit 2, when you compare two chunking strategies, index the second
    one as variant="v2" and you can query both instead of deleting the first
    and starting over.
    """
    name = config.collection_name(corpus, variant)
    client = _client()

    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        # ⚠️ Do not remove. Chroma defaults to squared L2, and every distance
        # number in this course assumes cosine.
        metadata={"hnsw:space": "cosine"},
    )

    batch = 256
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        collection.add(
            ids=[f"{c.source}#{c.index}" for c in window],
            documents=[c.text for c in window],
            embeddings=embed([c.text for c in window]),
            metadatas=[
                {"source": c.source, "index": c.index, "produced_by": c.produced_by}
                for c in window
            ],
        )

    return len(chunks)


def _result(text: str, meta: dict, distance: float) -> Result:
    return Result(
        text=text,
        source=str(meta.get("source", "unknown")),
        label=f"{meta.get('source', 'unknown')}#{meta.get('index', 0)}",
        distance=float(distance),
        produced_by=str(meta.get("produced_by", "unknown")),
    )


def search(
    question: str,
    top_k: int | None = None,
    corpus: str | None = None,
    variant: str = "default",
    mode: str | None = None,
) -> list[Result]:
    """
    Retrieve the chunks most relevant to a question.

    `mode` is "semantic" or "hybrid"; left out, it follows config.RETRIEVAL.
    Semantic results come back nearest-first. Hybrid results come back in
    fused order, which is not always nearest-first — but every result still
    carries its cosine distance, so the gate works the same either way.
    """
    top_k = top_k or config.TOP_K
    mode = mode or config.RETRIEVAL
    name = config.collection_name(corpus, variant)

    try:
        collection = _client().get_collection(name)
    except Exception as exc:
        raise RuntimeError(
            f"No index called '{name}'. Run `python app.py index` first."
        ) from exc

    if mode == "hybrid":
        return _hybrid_search(question, top_k, collection)
    if mode != "semantic":
        raise ValueError(f"Unknown retrieval mode {mode!r}: use 'semantic' or 'hybrid'.")

    raw = collection.query(
        query_embeddings=embed([question]),
        n_results=min(top_k, collection.count()),
    )

    return [
        _result(text, meta, distance)
        for text, meta, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    ]


# ─── Hybrid search: BM25 + embeddings ────────────────────────────────────────
#
# Embeddings match meaning; BM25 matches words. They fail differently:
# embeddings blur "CS 210" and "CS 340" together because they *mean* similar
# things, while BM25 has no idea "dorm" and "residence hall" are related.
# Running both and fusing the two rankings covers each one's blind spot.

# Words so common they'd only add noise to keyword matching. BM25's IDF already
# down-weights them, but question phrasing ("what do students say about...")
# is heavy with them, so dropping them up front keeps scores readable.
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its "
    "me my of on or so that the their there they this to was what when where "
    "which who why will with you your should would could about any".split()
)

# One BM25 index per collection, built lazily from what's already in Chroma so
# it always matches the chunks you indexed. Keyed on name and chunk count so a
# re-index in the same process is noticed.
_bm25_cache: dict[str, tuple[int, object, list[str]]] = {}


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def _bm25_for(collection):
    from rank_bm25 import BM25Okapi

    count = collection.count()
    cached = _bm25_cache.get(collection.name)
    if cached is not None and cached[0] == count:
        return cached[1], cached[2]

    data = collection.get(include=["documents"])
    bm25 = BM25Okapi([_tokenize(doc) for doc in data["documents"]])
    _bm25_cache[collection.name] = (count, bm25, data["ids"])
    return bm25, data["ids"]


def _cosine_distance(a, b) -> float:
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0


def _hybrid_search(question: str, top_k: int, collection) -> list[Result]:
    count = collection.count()
    if count == 0:
        return []
    pool = min(count, max(config.HYBRID_CANDIDATES, top_k))
    question_vector = embed([question])

    # Semantic side: the nearest `pool` chunks, with their real distances.
    raw = collection.query(query_embeddings=question_vector, n_results=pool)
    found = {
        chunk_id: (text, meta, float(distance))
        for chunk_id, text, meta, distance in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    }
    semantic_ranking = raw["ids"][0]

    # Keyword side: the `pool` best BM25 scores, ignoring chunks that share no
    # words with the question at all.
    bm25, ids = _bm25_for(collection)
    scores = bm25.get_scores(_tokenize(question))
    bm25_by_id = {chunk_id: float(s) for chunk_id, s in zip(ids, scores)}
    keyword_ranking = [
        chunk_id
        for chunk_id in sorted(bm25_by_id, key=bm25_by_id.get, reverse=True)[:pool]
        if bm25_by_id[chunk_id] > 0
    ]

    # Reciprocal Rank Fusion: each list votes 1 / (k + rank). Uses ranks, not
    # raw scores, because cosine distances and BM25 scores aren't on the same
    # scale and can't be added directly.
    fused: dict[str, float] = {}
    for ranking in (semantic_ranking, keyword_ranking):
        for rank, chunk_id in enumerate(ranking, 1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)
    winners = sorted(fused, key=fused.get, reverse=True)[:top_k]

    # Chunks BM25 pulled in that the semantic side never saw still need a
    # cosine distance, or the gate would have nothing to judge them by.
    missing = [chunk_id for chunk_id in winners if chunk_id not in found]
    if missing:
        extra = collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
        for chunk_id, text, meta, vector in zip(
            extra["ids"], extra["documents"], extra["metadatas"], extra["embeddings"]
        ):
            found[chunk_id] = (text, meta, _cosine_distance(question_vector[0], vector))

    results = []
    for chunk_id in winners:
        text, meta, distance = found[chunk_id]
        result = _result(text, meta, distance)
        result.bm25_score = bm25_by_id.get(chunk_id, 0.0)
        result.fused_score = fused[chunk_id]
        results.append(result)
    return results


def index_exists(corpus: str | None = None, variant: str = "default") -> bool:
    """Is there an index here to search, without searching it?

    `serve.py`'s health check asks this. It deliberately does not embed
    anything: loading the embedding model takes 80 MB and a few seconds, and a
    health check that heavy is a health check nobody can afford to call.
    """
    try:
        collection = _client().get_collection(config.collection_name(corpus, variant))
        return collection.count() > 0
    except Exception:
        return False


def reset():
    """Delete every index. Occasionally the fastest way out of a mess."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
