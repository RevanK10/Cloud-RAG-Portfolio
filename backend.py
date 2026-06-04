import os
import uuid
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

# 1. Try to load local variables if they exist
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
    "embedding-001",
]

_WORKING_GENERATION_MODEL = None
_GENERATION_MODEL_CANDIDATES = [
    os.getenv("GENERATION_MODEL", "").strip(),
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]


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
    """Lazily initialize cloud clients after secrets are available."""
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
        ai_client = None
        pc = None
        index = None
        return False


def get_index_stats():
    """UPGRADE: Fetches real-time vector counts directly from the Pinecone Cloud cluster."""
    if not _ensure_clients():
        return {"total_vector_count": 0}
    try:
        return index.describe_index_stats()
    except Exception:
        return {"total_vector_count": 0}


def clear_vector_database():
    """UPGRADE: Wipes out all stored vectors inside the index to reset the workspace."""
    if not _ensure_clients():
        return False
    try:
        index.delete(delete_all=True)
        return True
    except Exception:
        return False


def _discover_generation_models():
    if not _ensure_clients():
        return []
    try:
        discovered = []
        for m in ai_client.models.list():
            name = getattr(m, "name", None) or ""
            supported = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if "generateContent" in supported:
                short = name.removeprefix("models/") if name.startswith("models/") else name
                if "gemini" in name.lower():
                    discovered.extend([short, name])
        return list(dict.fromkeys(discovered))
    except Exception:
        return []


def _extract_embedding_values(response):
    if hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values
    if hasattr(response, "data") and response.data:
        first = response.data[0]
        if isinstance(first, dict):
            return first.get("embedding")
        if hasattr(first, "embedding"):
            return first.embedding
    raise ValueError("Unable to extract embedding values from response")


def _extract_text_match(match):
    metadata = None
    if isinstance(match, dict):
        metadata = match.get("metadata")
    elif hasattr(match, "metadata"):
        metadata = match.metadata

    if not metadata:
        return None

    if isinstance(metadata, dict):
        return metadata.get("text")
    return getattr(metadata, "text", None)


def _extract_matches(results):
    if isinstance(results, dict):
        return results.get("matches", [])
    return getattr(results, "matches", []) or []


def _iter_embedding_models():
    seen = set()
    ordered = []
    if _WORKING_EMBEDDING_MODEL:
        ordered.append(_WORKING_EMBEDDING_MODEL)
    ordered.extend(_EMBEDDING_MODEL_CANDIDATES)

    for model in ordered:
        if not model or model in seen:
            continue
        seen.add(model)
        yield model


def _embed_text(text: str):
    global _WORKING_EMBEDDING_MODEL
    last_error = None
    for model_name in _iter_embedding_models():
        try:
            response = ai_client.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
            )
            _WORKING_EMBEDDING_MODEL = model_name
            return _extract_embedding_values(response)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("No supported embedding model found.") from last_error


def _iter_generation_models():
    seen = set()
    ordered = []
    if _WORKING_GENERATION_MODEL:
        ordered.append(_WORKING_GENERATION_MODEL)
    env_model = os.getenv("GENERATION_MODEL", "").strip()
    if env_model:
        ordered.append(env_model)
    ordered.extend(_discover_generation_models())
    ordered.extend(_GENERATION_MODEL_CANDIDATES)

    for model in ordered:
        if not model or model in seen:
            continue
        seen.add(model)
        yield model


def _generate_text(prompt: str):
    global _WORKING_GENERATION_MODEL
    last_error = None
    for model_name in _iter_generation_models():
        try:
            response = ai_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            _WORKING_GENERATION_MODEL = model_name
            if response and getattr(response, "text", None):
                return response.text
            return "I cannot find that in the documents."
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("No supported generation model found.") from last_error


def upload_text_to_cloud(text: str, chunk_size: int = 600, chunk_overlap: int = 100):
    """UPGRADE: Slices text dynamically with sliding window character limits."""
    if not _ensure_clients() or not text.strip():
        return False

    # Prevent non-positive step sizes when overlap >= chunk size.
    safe_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    step = max(1, chunk_size - safe_overlap)

    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    
    try:
        for i, chunk in enumerate(chunks):
            vector = _embed_text(chunk)
            vector_id = f"vec_{int(time.time())}_{uuid.uuid4().hex[:6]}_{i}"
            index.upsert(vectors=[{"id": vector_id, "values": vector, "metadata": {"text": chunk}}])
        return True
    except Exception:
        return False


def query_rag_system(user_query: str):
    """UPGRADE: Queries Pinecone for context and returns a tuple (answer, source_chunks)."""
    if not _ensure_clients():
        return "Backend services are initializing or missing API keys.", []
        
    try:
        query_vector = _embed_text(user_query)
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)

        context_chunks = []
        for match in _extract_matches(results):
            chunk = _extract_text_match(match)
            if chunk:
                context_chunks.append(chunk)

        if not context_chunks:
            return "I cannot find that in the documents.", []

        context = "\n\n".join(context_chunks)
        
        prompt = (
            f"Answer the user's question using ONLY the provided context below.\n"
            f"If the answer is not in the context, say 'I cannot find that in the documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}"
        )
        
        answer = _generate_text(prompt)
        return answer, context_chunks
    except Exception as e:
        return f"Error executing RAG workflow: {str(e)}", []