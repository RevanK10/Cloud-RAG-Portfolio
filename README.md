# RAG Assistant

A Retrieval-Augmented Generation algorithm that allows the user to upload text documents and images to find more information about them.

The difference between this project and other default guides is that this setup is run mostly on the cloud. Instead of downloading heavy models, the algorithm uses a small local embedding model and larger online models to have higher performance using less local storage.

---

## Core Features

* **Reads Images and Visuals:** Users can drop text files or upload images directly into the sidebar to be parsed using the cloud model.
* **Smart processing:** This algorithm uses a combination of local storage and the cloud to handle all tasks. It uses local storage to handle embeddings, while using the internet to handle the rest of the processes.
* **Provides Sources:** The UI allows the user to view the area of the document where the algorithm found the answers.
* **Follow-Up Questions Suggested:** The UI suggests follow-up questions at the bottom of the screen.

---

## Technical Architecture
The workspace coordinates decoupled services to handle text parsing, geometric vector lookups, and conversational syntax alignment:

1. **File Consumption:** When the user uploads a file, the algorithm turns the file into text and sends the pictures of the text to an online system.
2. **Processing Stage:** The algorithm converts the long text into smaller pieces and turns them into a language that computers can understand.
3. **Storage:** The generated arrays are stored online so they can be accessed quickly later.
4. **Answer Generation:** When the user asks a question, the algorithm finds the most relevant pieces of text and returns them to an AI model to send an answer back to the user.

---

## Project Structure

- .env               # Confidential keys - Not in GitHub for security
- app.py             # UI - How the screen works and looks
- backend.py         # Handles the logic behind the answer generation
- OneTimeScript.py   # An algorithm that should be run once to set up the Pinecone index when using locally.
- Dockerfile         # Allows the algorithm to run on any computer
- requirements.txt   # Dependencies for the algorithm to run

---

## Setup and Installation

### 1. Secrets Configuration
Create a .env file in the root directory and enter your API credentials:

PINECONE_API_KEY="your-pinecone-api-key"
GEMINI_API_KEY="your-gemini-api-key"
PINECONE_INDEX="rag-portfolio"

### 2. Local Installation

pip install -r requirements.txt
python OneTimeScript.py
streamlit run app.py
