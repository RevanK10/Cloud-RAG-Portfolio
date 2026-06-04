---
title: Cloud RAG Assistant with Multimodal OCR
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Cloud-Powered Multimodal RAG Assistant

A production-grade, zero-local-storage Retrieval-Augmented Generation (RAG) workspace capable of indexing text documentation and transcribing handwritten notes on the fly. 

This architecture completely bypasses heavy local execution layers and bulky machine learning weights by offloading all embedding generation, visual character extraction (OCR), vector index storage, and language modeling to high-performance, cloud-native APIs.

---

##  Why This Project Exists (and The Upgrades)

Most standard RAG tutorials rely on downloading gigabytes of local model checkpoints (like Hugging Face or Ollama) or spinning up heavy local containerized databases. 

This project proves that an enterprise-ready, cross-modal knowledge engine can operate with a **0 MB local storage footprint**. By shifting heavy compute states to serverless infrastructure, the application environment acts purely as a low-latency traffic controller. 

### Core Features & Architectural Upgrades:
* ** Multimodal Visual Ingestion (OCR):** Bypasses basic copy-pasting. Users can drag-and-drop text files, upload images (`.png`, `.jpg`), or capture handwritten notes live using their smartphone or webcam.
* ** Glassmorphic Chat Workspace:** A polished, modern responsive front end using Streamlit styled with dynamic CSS gradient backdrops.
* ** Real-Time Cloud Telemetry:** Integrates directly with Pinecone control planes to report live, exact vector chunk numbers in the sidebar dashboard.
* ** Dynamic Slicing Matrix:** A dual-slider character-based sliding-window algorithm that prevents text truncation and allows manual drift tuning (Chunk Size vs. Overlap Buffer).
* ** Retained Source Receipts:** Complete transparency into model output. The interface highlights and isolates the exact context snippets retrieved from the cloud database to prove the answer is grounded.

---

## Technical Architecture

The system coordinates decoupled, cloud-native microservices to handle text parsing, geometric vector lookups, and conversational syntax alignment:

1. **The Ingestion Layer:** Plain text or images are accepted. Images are parsed by `gemini-2.5-flash` using native vision capabilities to extract structured typography.
2. **The Vector Processing Pipeline:** Extracted text is sliced dynamically using character buffers and mapped to high-density dense vectors ($768$ dimensions) via the Google Gen AI `text-embedding-004` API.
3. **The Storage Cluster:** Generated arrays are fed directly as structured metadata tuples into a **Pinecone Serverless Index**.
4. **The RAG Generation Loop:** Queries extract top-$k$ nearest neighbors from Pinecone, compile them into an immutable prompt layout, and pass them to `gemini-1.5-flash` to generate strict, zero-hallucination responses.

---

## Project Structure

This repository uses a strict **separation of concerns** strategy to keep UI layers separate from business data infrastructure logic:

```text
cloud-rag-portfolio
 ┣ .env                # Private cloud credentials (git-ignored)
 ┣ app.py              # Front-end design, UI state, layout tabs & styling (Streamlit)
 ┣ backend.py          # Cloud connector, Gemini SDK clients & Pinecone control loops
 ┣ Dockerfile          # Multi-stage production container blueprint
 ┗ requirements.txt    # Frozen, lightweight cloud client dependencies
