import os
import uuid
import time
import pypdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

load_dotenv()

ai_client = None
pc = None
index = None

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
_WORKING_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_CANDIDATES = [
    os.getenv("EMBEDDING_MODEL", "").strip(),
    "text-embedding-004",
    "gemini-embedding-001",
]

_WORKING_GENERATION_MODEL = None
_GENERATION_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

_PINECONE_BATCH_SIZE = 100  # Pinecone max upsert batch size


# ── Client bootstrap ──────────────────────────────────────────────────────────

def _get_secret(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        secret_value = st.secrets.get(name)
        if secret_value:
            os.environ[name] = str(secret_value)
            return str(secret_value)
    except Exception:
        return None
    return None


def _ensure_clients():
    global ai_client, pc, index
    if ai_client and index:
        return True

    google_key = _get_secret("GOOGLE_API_KEY")
    pinecone_key = _get_secret("PINECONE_API_KEY")
    index_name = _get_secret("PINECONE_INDEX") or "rag-portfolio"

    if not google_key or not pinecone_key:
        return False

    try:
        ai_client = genai.Client(api_key=google_key)
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        return True
    except Exception:
        return False


# ── Index utilities ───────────────────────────────────────────────────────────

def get_index_stats():
    if not _ensure_clients():
        return {"total_vector_count": 0}
    try:
        return index.describe_index_stats()
    except Exception:
        return {"total_vector_count": 0}


def get_indexed_sources() -> list[dict]:
    """
    Return a deduplicated list of source files currently in the index,
    with chunk counts. Uses a broad metadata fetch via a zero-vector query
    across all namespaces (serverless-compatible).
    """
    if not _ensure_clients():
        return []
    try:
        # Zero vector just to probe metadata; top_k gives us a sample
        zero = [0.0] * EMBEDDING_DIMENSION
        results = index.query(vector=zero, top_k=10000, include_metadata=True)
        matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", []) or []

        source_map: dict[str, dict] = {}
        for m in matches:
            meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)
            if not meta:
                continue
            src = meta.get("source", "Unknown")
            ts = float(meta.get("timestamp", 0))
            if src not in source_map:
                source_map[src] = {"source": src, "chunks": 0, "uploaded": ts}
            source_map[src]["chunks"] += 1
            source_map[src]["uploaded"] = max(source_map[src]["uploaded"], ts)

        return sorted(source_map.values(), key=lambda x: x["uploaded"], reverse=True)
    except Exception:
        return []


def delete_source(source_name: str) -> bool:
    """Delete all vectors belonging to a specific source file."""
    if not _ensure_clients():
        return False
    try:
        # Fetch IDs for this source
        zero = [0.0] * EMBEDDING_DIMENSION
        results = index.query(
            vector=zero, top_k=10000, include_metadata=True,
            filter={"source": {"$eq": source_name}}
        )
        matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", []) or []
        ids = [
            (m.get("id") if isinstance(m, dict) else getattr(m, "id", None))
            for m in matches
        ]
        ids = [i for i in ids if i]
        if ids:
            index.delete(ids=ids)
        return True
    except Exception:
        return False


def clear_vector_database() -> bool:
    if not _ensure_clients():
        return False
    try:
        index.delete(delete_all=True)
        return True
    except Exception:
        return False


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_wrapper) -> str:
    try:
        reader = pypdf.PdfReader(file_wrapper)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def process_image_ocr(image_bytes: bytes, mime_type: str) -> str:
    if not _ensure_clients():
        return ""
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                "Transcribe all visible text exactly as written. Include all structural elements, tables, and lists. Format clearly without commentary.",
            ],
        )
        return response.text if response.text else ""
    except Exception:
        return ""


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_text(text: str) -> list[float]:
    global _WORKING_EMBEDDING_MODEL
    if not _ensure_clients():
        raise RuntimeError("API clients not initialised.")

    ordered = [_WORKING_EMBEDDING_MODEL] if _WORKING_EMBEDDING_MODEL else []
    for c in _EMBEDDING_MODEL_CANDIDATES:
        if c and c not in ordered:
            ordered.append(c)

    last_error = None
    for model_name in ordered:
        if not model_name:
            continue
        try:
            response = ai_client.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
            )
            _WORKING_EMBEDDING_MODEL = model_name
            if hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0].values
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Embedding failed. Last error: {last_error}")


# ── Ingestion ─────────────────────────────────────────────────────────────────

def upload_text_to_cloud(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    source_file: str = "Unknown",
    progress_callback=None,
) -> int:
    """
    Chunk, embed, and upsert text into Pinecone.
    - Uses batch upserts (100 vectors per call) instead of one-by-one.
    - Calls progress_callback(done, total) if provided.
    Returns the number of chunks indexed, or 0 on failure.
    """
    if not _ensure_clients() or not text.strip():
        return 0

    safe_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    step = max(1, chunk_size - safe_overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    if not chunks:
        return 0

    timestamp = str(time.time())
    vectors_to_upsert: list[dict] = []

    try:
        for i, chunk in enumerate(chunks):
            vec = _embed_text(chunk)
            vectors_to_upsert.append({
                "id": f"vec_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}",
                "values": vec,
                "metadata": {
                    "text": chunk,
                    "source": source_file,
                    "timestamp": timestamp,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                },
            })
            if progress_callback:
                progress_callback(i + 1, len(chunks))

        # Batch upsert — dramatically faster and fewer API calls
        for batch_start in range(0, len(vectors_to_upsert), _PINECONE_BATCH_SIZE):
            batch = vectors_to_upsert[batch_start : batch_start + _PINECONE_BATCH_SIZE]
            index.upsert(vectors=batch)

        return len(chunks)
    except Exception:
        return 0


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _rewrite_query_with_history(user_query: str, history: list) -> str:
    """
    If the conversation has history, ask Gemini to rewrite the user's query
    into a fully self-contained retrieval query. This fixes follow-up questions
    like 'tell me more about that' which are meaningless to a vector search.
    """
    if not history or not _ensure_clients():
        return user_query

    # Only bother if the query is short / vague (likely a follow-up)
    if len(user_query.split()) > 10:
        return user_query

    recent = []
    for msg in history[-4:]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:400]
        if content:
            recent.append(f"{role}: {content}")

    prompt = (
        "You are a query rewriter for a retrieval system.\n"
        "Given the conversation history and the latest user message, "
        "rewrite the latest message as a single, fully self-contained search query "
        "that captures the user's true intent. Output ONLY the rewritten query, nothing else.\n\n"
        f"History:\n" + "\n".join(recent) + f"\n\nLatest message: {user_query}\n\nRewritten query:"
    )

    model = _WORKING_GENERATION_MODEL or _GENERATION_MODEL_CANDIDATES[0]
    try:
        response = ai_client.models.generate_content(model=model, contents=prompt)
        rewritten = (response.text or "").strip().strip('"')
        return rewritten if rewritten else user_query
    except Exception:
        return user_query


def _rerank_chunks(user_query: str, chunks: list[dict]) -> list[dict]:
    """
    Re-rank retrieved chunks by asking Gemini to score each one for
    relevance to the query. Falls back to original order on error.
    Only runs when there are enough chunks to be worth reranking.
    """
    if len(chunks) <= 2 or not _ensure_clients():
        return chunks

    numbered = "\n\n".join(
        f"[{i}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"You are a relevance scorer. Given the query and {len(chunks)} text chunks, "
        f"output ONLY a JSON array of indices sorted by relevance (most relevant first).\n"
        f"Example for 3 chunks: [2,0,1]\n\n"
        f"Query: {user_query}\n\nChunks:\n{numbered}\n\nSorted indices:"
    )

    model = _WORKING_GENERATION_MODEL or _GENERATION_MODEL_CANDIDATES[0]
    try:
        response = ai_client.models.generate_content(model=model, contents=prompt)
        raw = (response.text or "").strip()
        # Extract the JSON array
        import re, json
        match = re.search(r"\[[\d,\s]+\]", raw)
        if match:
            indices = json.loads(match.group())
            reranked = []
            seen = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                    reranked.append(chunks[idx])
                    seen.add(idx)
            # Append any missing chunks at the end
            for i, c in enumerate(chunks):
                if i not in seen:
                    reranked.append(c)
            return reranked
    except Exception:
        pass
    return chunks


def query_rag_system_context(user_query: str, history: list = None) -> tuple[str, list]:
    """
    Full retrieval pipeline:
    1. Rewrite query with conversation context
    2. Embed and retrieve from Pinecone
    3. Re-rank by relevance
    Returns (context_string, source_receipts)
    """
    if not _ensure_clients():
        return "", []

    try:
        # Step 1: query rewriting for follow-up questions
        retrieval_query = _rewrite_query_with_history(user_query, history or [])

        # Step 2: vector retrieval
        query_vector = _embed_text(retrieval_query)
        results = index.query(vector=query_vector, top_k=7, include_metadata=True)
        matches = (
            results.get("matches", []) if isinstance(results, dict)
            else getattr(results, "matches", []) or []
        )

        raw_chunks = []
        for match in matches:
            meta = (
                match.get("metadata") if isinstance(match, dict)
                else getattr(match, "metadata", None)
            )
            score = (
                match.get("score", 0.0) if isinstance(match, dict)
                else getattr(match, "score", 0.0)
            )
            if meta and isinstance(meta, dict) and meta.get("text"):
                raw_chunks.append({
                    "text": meta["text"],
                    "source": meta.get("source", "Unknown"),
                    "score": float(score),
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks": meta.get("total_chunks", 0),
                })

        if not raw_chunks:
            return "", []

        # Step 3: re-rank
        reranked = _rerank_chunks(user_query, raw_chunks)

        context_chunks = []
        source_receipts = []
        for chunk in reranked[:5]:  # use top 5 after reranking
            context_chunks.append(chunk["text"])
            source_receipts.append({
                "file": chunk["source"],
                "text": chunk["text"],
                "confidence": f"{chunk['score'] * 100:.1f}%",
                "chunk": f"{chunk['chunk_index'] + 1}/{chunk['total_chunks']}" if chunk["total_chunks"] else "—",
            })

        return "\n\n".join(context_chunks), source_receipts

    except Exception:
        return "", []


# ── Generation ────────────────────────────────────────────────────────────────

def generate_follow_up_questions(user_query: str, answer: str, context: str) -> list[str]:
    """Generate 3 smart follow-up questions based on the answer and context."""
    if not _ensure_clients():
        return []

    prompt = (
        "Based on this Q&A exchange and the underlying document context, "
        "generate exactly 3 concise, specific follow-up questions the user might want to ask next. "
        "Output ONLY a JSON array of 3 question strings, nothing else.\n\n"
        f"User asked: {user_query}\n"
        f"Answer summary: {answer[:500]}\n"
        f"Context excerpt: {context[:600]}\n\n"
        "Follow-up questions (JSON array):"
    )

    model = _WORKING_GENERATION_MODEL or _GENERATION_MODEL_CANDIDATES[0]
    try:
        response = ai_client.models.generate_content(model=model, contents=prompt)
        raw = (response.text or "").strip()
        import re, json
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            return [q for q in questions if isinstance(q, str)][:3]
    except Exception:
        pass
    return []


def self_correct_response(user_query: str, proposed_answer: str, context: str) -> str:
    """Faithfulness check — rewrites the answer if it contains hallucinations."""
    if not _ensure_clients():
        return proposed_answer

    _cannot_find_signals = [
        "cannot find that in the documents",
        "i cannot find",
        "not found in the",
        "no relevant information",
        "no information available",
    ]
    if any(sig in proposed_answer.lower() for sig in _cannot_find_signals):
        return proposed_answer

    verification_prompt = (
        "You are a Quality Verification Agent.\n"
        "Review the Proposed Answer against the Document Context for absolute faithfulness.\n"
        "Rules:\n"
        "- If the answer is fully supported by the context, return it verbatim.\n"
        "- If it contains ANY detail not in the context, rewrite it to be 100% faithful.\n"
        "- Never add information beyond what the context contains.\n"
        "- Preserve markdown formatting.\n\n"
        f"Document Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        f"Proposed Answer:\n{proposed_answer}\n\n"
        "Verified response (no meta-commentary):"
    )

    model = _WORKING_GENERATION_MODEL or _GENERATION_MODEL_CANDIDATES[0]
    try:
        response = ai_client.models.generate_content(model=model, contents=verification_prompt)
        return response.text.strip() if response.text else proposed_answer
    except Exception:
        return proposed_answer


def generate_streaming_chat_response(user_query: str, context: str, history: list):
    """
    Streams the answer token-by-token.
    Persists the working model so subsequent calls skip the candidate loop.
    History 'sources' and other extra keys are safely ignored.
    """
    global _WORKING_GENERATION_MODEL

    if not _ensure_clients():
        yield "⚠️ API clients disconnected. Please check your API keys."
        return

    history_context = ""
    if history:
        safe_turns = []
        for msg in history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            if content:
                safe_turns.append(f"{role}: {content}")
        if safe_turns:
            history_context = "Previous Conversation:\n" + "\n".join(safe_turns) + "\n\n"

    system_prompt = (
        "You are an expert enterprise knowledge assistant with deep analytical capability.\n"
        "Answer using ONLY the provided context. Be precise, well-structured, and thorough.\n"
        "Use markdown formatting where it improves clarity: headers, bullets, bold terms, code blocks.\n"
        "If the answer is not in the context, state exactly: 'I cannot find that in the documents.'\n\n"
        f"{history_context}"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}"
    )

    ordered = []
    if _WORKING_GENERATION_MODEL:
        ordered.append(_WORKING_GENERATION_MODEL)
    for m in _GENERATION_MODEL_CANDIDATES:
        if m and m not in ordered:
            ordered.append(m)

    last_error = None
    for model_name in ordered:
        for attempt in range(3):
            try:
                response_stream = ai_client.models.generate_content_stream(
                    model=model_name,
                    contents=system_prompt,
                )
                token_count = 0
                for chunk in response_stream:
                    if chunk.text:
                        token_count += 1
                        yield chunk.text
                if token_count > 0:
                    _WORKING_GENERATION_MODEL = model_name
                    return
                break  # zero tokens, no exception — soft failure, try next model
            except Exception as e:
                last_error = e
                time.sleep(0.5 * (2 ** attempt))
                continue

    yield (
        f"\n\n⚠️ **Generation temporarily unavailable.** "
        f"Please retry. *(Error: {type(last_error).__name__}: {last_error})*"
    )