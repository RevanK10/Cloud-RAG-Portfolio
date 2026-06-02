import os
import uuid
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

# 1. Try to load local variables if they exist (local testing)
load_dotenv()

# 2. Grab keys securely (works both locally from .env AND on Hugging Face from Secrets)
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

# 3. Prevent crashing at boot time if keys are loading slowly
if GOOGLE_KEY and PINECONE_KEY:
    ai_client = genai.Client(api_key=GOOGLE_KEY)
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index("rag-portfolio")
else:
    ai_client = None
    pc = None
    index = None

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
_WORKING_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_CANDIDATES = [
    os.getenv("EMBEDDING_MODEL", "").strip(),
    "gemini-embedding-001",
    "text-embedding-004",
    "embedding-001",
    "models/gemini-embedding-001",
    "models/text-embedding-004",
    "models/embedding-001",
]


def _extract_embedding_values(response):
    """Supports multiple google-genai response shapes."""
    if hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values

    # Fallback for alternate response shape used by some SDK versions.
    if hasattr(response, "data") and response.data:
        first = response.data[0]
        if isinstance(first, dict):
            return first.get("embedding")
        if hasattr(first, "embedding"):
            return first.embedding

    raise ValueError("Unable to extract embedding values from response")


def _extract_text_match(match):
    """Reads metadata text from either Pinecone objects or dicts."""
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
    """Reads query matches from either Pinecone objects or dicts."""
    if isinstance(results, dict):
        return results.get("matches", [])
    return getattr(results, "matches", []) or []


def _iter_embedding_models():
    """Yield candidate embedding model names, preferring previously working model."""
    seen = set()
    ordered = []
    if _WORKING_EMBEDDING_MODEL:
        ordered.append(_WORKING_EMBEDDING_MODEL)
    ordered.extend(_EMBEDDING_MODEL_CANDIDATES)

    for model in ordered:
        if not model:
            continue
        if model in seen:
            continue
        seen.add(model)
        yield model


def _embed_text(text: str):
    """Try supported embedding models and cache the first one that works."""
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

    raise RuntimeError(
        "No supported embedding model found. "
        "Set EMBEDDING_MODEL in environment to a valid embedding model for your API key."
    ) from last_error

def upload_text_to_cloud(text: str):
    """Slices text dynamically and uploads vectors directly to Pinecone Cloud."""
    if not ai_client or not index:
        return False
    if not text.strip():
        return False
        
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    for i, chunk in enumerate(paragraphs):
        vector = _embed_text(chunk)
        vector_id = f"vec_{uuid.uuid4().hex}_{i}"
        index.upsert(vectors=[{"id": vector_id, "values": vector, "metadata": {"text": chunk}}])
    return True

def query_rag_system(user_query: str):
    """Queries Pinecone for context and uses Gemini to answer."""
    if not ai_client or not index:
        return "Backend services are initializing or missing API keys. Check your Space Secrets."
        
    query_vector = _embed_text(user_query)
    
    results = index.query(vector=query_vector, top_k=2, include_metadata=True)

    context_chunks = []
    for match in _extract_matches(results):
        chunk = _extract_text_match(match)
        if chunk:
            context_chunks.append(chunk)

    context = "\n\n".join(context_chunks)
    
    prompt = (
        f"Answer the user's question using ONLY the provided context below.\n"
        f"If the answer is not in the context, say 'I cannot find that in the documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}"
    )
    
    llm_response = ai_client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt,
    )
    return llm_response.text if llm_response and llm_response.text else "I cannot find that in the documents."