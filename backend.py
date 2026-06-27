import os
import uuid
import base64
import time
import re
import pypdf
from pinecone import Pinecone
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=".env", override=True)
import streamlit as st


# ==========================
# ENV
# ==========================

def load_env():
    load_dotenv(override=True)

    pinecone_key = os.getenv("PINECONE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "rag-portfolio")

    if not pinecone_key:
        raise ValueError("Missing PINECONE_API_KEY")

    if not gemini_key:
        raise ValueError("Missing GEMINI_API_KEY")

    return pinecone_key, gemini_key, index_name


PINECONE_API_KEY, GEMINI_API_KEY, PINECONE_INDEX = load_env()

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

# ==========================
# GEMINI CLIENT
# ==========================

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================
# PINECONE
# ==========================

pc = Pinecone(api_key=PINECONE_API_KEY)

# Debug
if PINECONE_INDEX not in pc.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX}' does not exist.")

index = pc.Index(PINECONE_INDEX)

# ==========================
# PDF EXTRACTION
# ==========================

def extract_text_from_pdf(file):
    try:
        reader = pypdf.PdfReader(file)
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        print("PDF ERROR:", e)
        return ""

# ==========================
# OCR (GEMINI VISION)
# ==========================

def process_image_ocr(image_bytes):
    try:
        img_base64 = base64.b64encode(image_bytes).decode()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {"text": "Extract all visible text exactly."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        )

        return response.text

    except Exception as e:
        print("OCR ERROR:", e)
        return ""

# ==========================
# EMBEDDINGS (GEMINI)
# ==========================

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_text(text):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print("EMBED ERROR:", e)
        return []

# ==========================
# CHUNKING
# ==========================

def smart_chunk(text):
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []

    chunk_size = 150
    overlap = 30

    i = 0

    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

        i += (chunk_size - overlap)

    return chunks

# ==========================
# INDEXING
# ==========================

def upload_document(text, source):
    if not text.strip():
        return 0

    chunks = smart_chunk(text)
    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        if not embedding:
            continue

        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": source,
                "chunk": i
            }
        })

    if vectors:
        index.upsert(vectors=vectors)

    return len(vectors)

# ==========================
# RETRIEVAL
# ==========================

def retrieve(query):
    embedding = embed_text(query)

    if not embedding:
        print("EMPTY QUERY EMBEDDING")
        return []

    results = index.query(
        vector=embedding,
        top_k=10,
        include_metadata=True
    )

    matches = results.get("matches", [])
    matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)

    output = []

    for match in matches:
        metadata = match.get("metadata", {})

        text = metadata.get("text", "")
        if not text:
            continue

        output.append({
            "text": text,
            "score": match.get("score", 0),
            "source": metadata.get("source", "unknown")
        })

    return output


# ==========================
# ANSWER (GEMINI)
# ==========================

def generate_answer(query):
    retrieved = retrieve(query)

    if not retrieved:
        return "No relevant information found in your uploaded documents.", []

    context = "\n\n".join([x["text"] for x in retrieved])

    prompt = f"""
Use ONLY the context below.

If the answer is not in the context, say:
"I don't know based on the documents."

Context:
{context}

Question:
{query}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text, retrieved

    except Exception as e:
        print("ANSWER ERROR:", e)
        return "Error generating answer.", []

# ==========================
# FOLLOWUPS (GEMINI)
# ==========================

def generate_followups(query, answer):
    try:
        prompt = f"""
Generate 3 short useful follow-up questions.

Question: {query}
Answer: {answer}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw = response.text
        lines = [x.strip("- ").strip() for x in raw.split("\n") if x.strip()]

        return lines[:3]

    except Exception as e:
        print("FOLLOWUP ERROR:", e)
        return []