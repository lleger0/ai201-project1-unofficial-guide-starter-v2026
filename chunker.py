"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

import re
from dataclasses import dataclass

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split documents into chunks. ⚠️ REPLACE THE BODY OF THIS IN MILESTONE 3.

    Right now it just calls the fallback. That is the plain, generic behaviour
    the brief is talking about.

    When you write your own strategy, set `produced_by` to
    "chunker.py::split_documents" so your README's Sample Chunks section names
    the right function. `app.py chunks` prints that string for you.

    Things worth thinking about before you write any code:
      - Are your documents short posts or long guides?
      - Is the useful information in one sentence, or spread over a paragraph?
      - Would splitting on paragraph breaks keep more thoughts intact than
        splitting on a character count?
    """

    # These documents are forum threads: a question, then the replies to it.
    # A reply on its own is a poor chunk — "16 is the answer" has none of the
    # words a question about laptops would match against, and the model reading
    # it can't tell what was asked. So every chunk here is a pair: the thread's
    # question, plus exactly one reply to it.
    #
    # That means the question is repeated across every chunk from a thread,
    # which is the point. Each chunk still carries a different answer, and each
    # one can be retrieved and understood on its own.
    #
    # Chunks still break only at the end of a reply, or — for a reply that runs
    # long — at the end of a sentence, so nothing is cut mid-thought and there
    # is no overlap to paper over a bad cut.

    # Every response is introduced by a line like "--- reply 2 (21 votes) ---".
    # The group around it keeps that line in the split output, so each chunk can
    # be labelled with the reply it came from — the vote count included.
    reply_marker = re.compile(
        r"^([ \t]*-{2,}\s*reply\b.*?-{2,}[ \t]*)$", re.MULTILINE | re.IGNORECASE
    )
    # A sentence ends at . ! or ? — plus any closing quote or bracket — followed
    # by whitespace. Good enough for prose typed into a forum box.
    sentence_end = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
    # How much of one reply rides in a single chunk, before the question is
    # added on top. Most replies here are well under it and stay whole.
    max_reply_chars = 600

    chunks: list[Chunk] = []

    for doc in documents:
        index = 0

        # Split the thread at the reply markers. Because the pattern captures,
        # this comes back as [question, marker, reply, marker, reply, ...]: the
        # text before the first marker is the "THREAD: ..." question line, and
        # after it the markers and replies alternate.
        parts = reply_marker.split(doc.text)
        question = parts[0].strip()

        # Pair each marker with the reply that follows it.
        replies = [
            (parts[i].strip(), parts[i + 1].strip())
            for i in range(1, len(parts) - 1, 2)
            if parts[i + 1].strip()
        ]

        # A document with no reply markers (a guide, a single post) has neither
        # a question nor a marker to pair anything with, so it is chunked as it
        # stands. That keeps this working on the other corpora.
        if not replies:
            if not question:
                continue
            question, replies = "", [("", question)]

        for marker, reply in replies:
            # Short enough to stand as one chunk: the question and the whole
            # reply together.
            if len(reply) <= max_reply_chars:
                pieces = [reply]
            else:
                # Too long, so fall back to sentence boundaries: walk the
                # sentences and start a new piece whenever adding the next one
                # would push past the limit. A single sentence longer than the
                # limit is kept whole rather than cut in half. Every piece gets
                # the question and the marker, so each one still reads on its
                # own.
                sentences: list[str] = []
                start = 0
                for match in sentence_end.finditer(reply):
                    sentence = reply[start : match.end()].strip()
                    if sentence:
                        sentences.append(sentence)
                    start = match.end()
                tail = reply[start:].strip()
                if tail:
                    sentences.append(tail)

                pieces = []
                current = ""
                for sentence in sentences:
                    if current and len(current) + 1 + len(sentence) > max_reply_chars:
                        pieces.append(current)
                        current = sentence
                    else:
                        current = f"{current} {sentence}" if current else sentence
                if current:
                    pieces.append(current)

            for piece in pieces:
                # Lay the chunk out the way the source file does: the question,
                # a blank line, then the reply under its own marker line.
                prefix = f"{question}\n\n" if question else ""
                if marker:
                    prefix += f"{marker}\n"
                text = f"{prefix}{piece}"
                chunks.append(
                    Chunk(
                        text=text,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::split_documents",
                    )
                )
                index += 1

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
