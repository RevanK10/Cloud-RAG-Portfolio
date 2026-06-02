import streamlit as st
import os
from backend import upload_text_to_cloud, query_rag_system

# Page Configurations
st.set_page_config(page_title="Cloud RAG Portfolio", page_icon="🤖", layout="centered")
st.title("Cloud-Powered RAG Assistant")
st.write("Lightweight SDK Build: Zero Local Storage & Compilers Required")

# Guard rails for checking your space secrets
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please verify your local .env file exists or Secrets are set on Hugging Face.")
    st.stop()

# Sidebar Layout for data ingestion
with st.sidebar:
    st.header("Data Ingestion")
    st.write("Paste text below to update your Cloud Vector Database index.")

    user_text = st.text_area("Source Text:", height=200, placeholder="Paste text here...")
    process_button = st.button("Upload to Cloud DB", type="primary")

    if process_button:
        if user_text:
            with st.spinner("Uploading vectors to Pinecone cloud..."):
                if upload_text_to_cloud(user_text):
                    st.success("Successfully stored in Pinecone!")
                else:
                    st.error("Failed to upload. Backend services initializing or error occurred.")
        else:
            st.warning("Please enter some text before uploading.")
    
# Initialize chat log state tracking
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display entire history tracking on stream refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt ingestion logic (Fixed typo here)
if user_query := st.chat_input("Ask something about your uploaded data..."):
    # Display user input bubble instantly (Fixed scope issue here)
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Query processing through backend client
    with st.chat_message("assistant"):
        with st.spinner("Querying Cloud Systems..."):
            try:
                answer = query_rag_system(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error querying cloud backend: {e}")