# Cloud-Powered Multimodal RAG Assistant Atlas Workspace

A streamlined, cloud-integrated Retrieval-Augmented Generation workspace capable of indexing text documentation and transcribing images. 

While traditional setups require downloading gigabytes of local machine learning models or running heavy databases, this architecture shifts heavy processing and storage to cloud infrastructure. By offloading character extraction, vector storage, and language generation to high-performance cloud APIs, the core application environment functions as a low-latency controller.

---

## System Discrepancies Resolved

To ensure accurate documentation, several critical conflicts between the initial draft and the underlying code have been corrected:

* The system utilizes a local model for embedding generation. Because this downloads local weights via PyTorch, the system minimizes cloud transactions but does not maintain a strict zero local footprint.
* The index setup targets a 384-dimensional vector field matching the local transformer. It does not map to a 768-dimension layout or utilize text-embedding-004.
* The generation pipeline consistently connects to gemini-2.5-flash for both image parsing and context-grounded response composition.
* Features like dynamic sliders for chunking and live sidebar index telemetry are noted in design concepts but are currently handled via fixed code configurations in the backend.

---

## Core Features

* **Multimodal Visual Ingestion:** Users can drop text files or upload images directly into the sidebar to be parsed using the vision capabilities of the cloud model.
* **Hybrid Processing Pipeline:** Balancing cloud processing and localized computing, the system offloads orchestration to serverless models while managing text token embeddings locally.
* **Retained Source Receipts:** The interface displays a collapsed source layout showing the exact context chunks and document sources retrieved from the database.
* **Interactive Follow Ups:** Uses an automated loop to suggest relevant follow-up prompts beneath responses, allowing you to instantly expand your search depth with a single click.

---

## Technical Architecture

The workspace coordinates decoupled services to handle text parsing, geometric vector lookups, and conversational syntax alignment:

1. **The Ingestion Layer:** Documents are ingested via the user interface. PDF files are parsed via pypdf, while images are transferred as base64 byte sequences to the cloud model for explicit text extraction.
2. **The Vector Processing Pipeline:** Text is programmatically structured using a rolling sliding-window chunking algorithm set to 150-word blocks with a 30-word overlap. These text segments are mapped to dense vectors of 384 dimensions via a local embedding pipeline.
3. **The Storage Cluster:** Generated arrays are fed directly as structured metadata tuples into a Pinecone Serverless Index hosted on AWS.
4. **The RAG Generation Loop:** Input queries are vectorized, pulling the top 12 nearest neighbors from the database. These chunks are isolated into a restricted prompt layout inside gemini-2.5-flash to generate factual, document-grounded responses.

---

## Project Structure

This repository uses a strict separation of concerns strategy to keep UI layers separate from business data infrastructure logic:

```text
cloud-rag-portfolio
 ┣ .env                # Private cloud credentials - Not provided in GitHub
 ┣ app.py              # Front-end design, UI state, and custom CSS styling
 ┣ backend.py          # Embedding logic, cloud SDK clients, and database query loops
 ┣ OneTimeScript.py    # Independent utility script to provision and build the index
 ┣ Dockerfile          # Production container blueprint
 ┗ requirements.txt    # Frozen backend and front-end dependencies
```

---

## Setup and Installation

### 1. Environment Configuration
Create a `.env` file in the root directory and populate your API credentials:
```env
PINECONE_API_KEY="your-pinecone-api-key"
GEMINI_API_KEY="your-gemini-api-key"
PINECONE_INDEX="rag-portfolio"
```

### 2. Local Installation
```bash
pip install -r requirements.txt
python OneTimeScript.py
streamlit run app.py
```
README-v2.md
Displaying README-v2.md.
