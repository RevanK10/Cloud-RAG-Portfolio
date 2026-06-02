#Basic code for the RAG app - includes logic and UI
"""import os
from dotenv import load_dotenv
from langchain_text_splitters import RecusriveCharacterTextSplitter
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from lanchain.chains import create_retrieval_chain
from langchains.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

raw_text = ""
The Solar System formed 4.6 billiosn years ago from the gravitational collaps of a giant interstellar molecular cloud.
The vast majority of the system's mass is in the Sun, with the majority of the remaining mass contained in Jupiter.
The four smaller inner system planets, Mercury, Venus, Earth and Mars, are terrestrial planets, being primarily composed of rock and metal.
""

embeddins = GoogleGenAIEmbeddings(model="models/text-embedding-004")

index_name = "rag-portfolio"
vector_store = PineConeVectorStore.from_documents(docs, embeddings, index_name=index_name)
retriever = vector_store.as_retriever(search_kwargs={"k":2})

llm - ChatGoogleGenerativeAI(model="gemini-1.5-flash")

system_prompt = "Answer the question using only this context:\n\n{context}"
prompt = ChatPromptTemplate.from_messgae([
    ("system", system_prompt),
    {"human", "{input}"}
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_stuff_documents_chain(retriever, qa_chain)

query = "What are the inner planets made of?"
response = rag_chain.invoke({"input": query})

print("\n--- Cloud RAG Response ---")
print(response["answer"])
#UI for the RAG App
import streamlit as st
import os

from backend import upload_text_to_cloud, get_rag_chain

st.set_page_config(page_title="Cloud RAG Portfolio", page_icon="R", layout="centered")
st.title("Cloud-Powered RAG Assistant")
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please verify your local .env file exsists.")
    st.stop()

with st.sidebar:
    st.header("Data Ingestion")
    st.write("Paste text below to update your Cloud Vector Database index.")

    user_text= st.text_area("Source Text:", height=200, placeholder="Paste text here...")
    process_button = st.button("Upload to Cloud DB", type="primary")

    if process_button:
        if user_text:
            with st.spinner("Processing & uploading to Pinecone cloud..."):
                success = upload_text_to_cloud(user_text)
                if success:
                    st.success("Successfully stored in the cloud!")
        else:
            st.warning("Please enter text before uploading.")

@st.cache_resource
def load_chain():
    return get_rag_chain()

rag_chain = load_chain()

if "messahes" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_inpt("Ask about your data..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role":"user", "content":user_query})

    with st.chat_messahe("assistant"):
        with st.spinner("Querying Cloud DB & Gemini..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                output_text = response["answer"]
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"Error querying cloud backend: {e}")
"""


import streamlit as st
import os
from backend import upload_text_to_cloud, query_rag_system

st.set_page_config(page_title="Cloud RAG Portfolio", page_icon="R", layout="centered")
st.title("Cloud-Powered RAG Assistant")
st.write("Lightweight SDK Build: Zero Local Storage & Compilers Required")

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please verify your local .env file exsists.")
    st.stop()

with st.sidebar:
    st.header("Data Ingestion")
    st.write("Paste text below to update your Cloud Vector Database index.")

    user_text = st.text_area("Source Text:", height=200, placeholder="Paste text here...")
    process_button = st.button("Upload to Cloud DB", type="primary")

    if process_button:
        if user_text:
            with st.spinner("Uploading vectors to Pinecone cloud..."):
                if upload_text_to_cloud(user_text):
                    st.success("Successfully stores in Pinecone!")
        else:
            st.warning("Please enter some text before uploading.")
    
if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about your uploaded data..."):
    with st.chat_message(message["role"]):
        st.markdown(user_query)
    st.session_state.messages.append({"role":"user", "content":user_query})

    with st.chat_message("assistant"):
        with st.spinner("Querying Cloud Systems..."):
            try:
                answer = query_rag_system(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role":"assisstant", "content":answer})
            except Exception as e:
                st.error(f"Error querying cloud backend: {e}")
