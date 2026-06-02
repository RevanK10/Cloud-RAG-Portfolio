import os
from dotenv import load_dotenv
from google import genai
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

def upload_text_to_cloud(text: str):
    """Slices text dynamically and uploads vectors directly to Pinecone Cloud."""
    if not ai_client or not index:
        return False
    if not text.strip():
        return False
        
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    for i, chunk in enumerate(paragraphs):
        # FIX: Swapped legacy model for SDK-supported gemini-embedding-001
        response = ai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        vector = response.embeddings[0].values
        index.upsert(vectors=[(f"vec_{i}", vector, {"text": chunk})])
    return True

def query_rag_system(user_query: str):
    """Queries Pinecone for context and uses Gemini to answer."""
    if not ai_client or not index:
        return "Backend services are initializing or missing API keys. Check your Space Secrets."
        
    # FIX: Swapped legacy model for SDK-supported gemini-embedding-001
    response = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=user_query
    )
    query_vector = response.embeddings[0].values
    
    results = index.query(vector=query_vector, top_k=2, include_metadata=True)
    
    context_chunks = [match['metadata']['text'] for match in results['matches'] if 'metadata' in match]
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
    return llm_response.text