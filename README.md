
---
title: Cloud RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Cloud-Powered RAG Assistant

A lightweight, zero-local-storage Retrieval-Augmented Generation (RAG) chat application. 

This project bypasses heavy local dependencies and bulky machine learning libraries by routing all embedding generation, vector storage, and language modeling through native, free-tier cloud APIs. 

---

## Why This Project Exists

Most introductory RAG tutorials require downloading hundreds of megabytes of local model weights (like Hugging Face transformers) and setting up heavy databases on your hard drive. 

I built this version to prove that you can build a highly performant, conversational AI tool with **zero local storage footprint**. By leveraging official cloud SDKs directly, the local or hosted environment acts purely as a traffic controller, keeping the hardware footprint close to 0 MB.

### The Architecture:
* **The Frontend:** Built using Streamlit to create a native, conversational chat window and data upload dashboard.
* **The Vector DB:** Handled via **Pinecone's Cloud Free Tier**. Chunks are updated and queried completely in the cloud.
* **The Brain:** Handled via official **Google Gemini 1.5 Flash API** for processing deep text embeddings and generating conversational answers.

---

## Project Structure

The project uses a clean **separation of concerns** strategy, dividing the visual interface from the underlying technical logic:

```text
cloud-rag-portfolio
 ┣  .env                # Private cloud credentials (hidden from GitHub)
 ┣  app.py              # Frontend UI layer (Streamlit)
 ┣  backend.py          # Cloud logic layer (Gemini & Pinecone APIs)
 ┣  Dockerfile          # Server deployment instructions
 ┗  requirements.txt    # Lightweight project dependencies
