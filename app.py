import streamlit as st
import os
from backend import upload_text_to_cloud, query_rag_system

st.set_page_config(page_title="Cloud RAG Portfolio", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(0, 163, 255, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 28%),
                linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            color: #0f172a;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 2rem 2.25rem;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(16px);
            margin-bottom: 1.5rem;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: 0.9rem;
        }
        .hero h1 {
            font-size: clamp(2.3rem, 4vw, 4.2rem);
            line-height: 1.02;
            margin: 0;
            color: #0f172a;
        }
        .hero p {
            margin: 0.9rem 0 0;
            max-width: 60ch;
            color: #475569;
            font-size: 1.05rem;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }
        .stat-card, .panel-card {
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
        }
        .stat-card {
            padding: 1rem 1.1rem;
        }
        .stat-card span {
            display: block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.45rem;
        }
        .stat-card strong {
            display: block;
            font-size: 1.05rem;
            color: #0f172a;
        }
        .panel-card {
            padding: 1.25rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5eefb;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #e5eefb !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 0;
            background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
            color: white;
            font-weight: 700;
            padding: 0.8rem 1rem;
        }
        .stChatMessage {
            border-radius: 18px;
            padding: 0.25rem 0;
        }
        .stTextArea textarea {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.92);
        }
        .stChatInput textarea {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.95);
        }
        .hint {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Cloud-native RAG workspace</div>
        <h1>Cloud-Powered RAG Assistant</h1>
        <p>Upload text into a managed vector index and ask questions in a polished, chat-first interface powered entirely by cloud services.</p>
        <div class="stat-grid">
            <div class="stat-card"><span>Deployment style</span><strong>Streamlit front end</strong></div>
            <div class="stat-card"><span>Storage model</span><strong>No local persistence</strong></div>
            <div class="stat-card"><span>Inference path</span><strong>Gemini + Pinecone</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard rails for checking your space secrets
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please verify your local .env file exists or Secrets are set on Hugging Face.")
    st.stop()

# Sidebar Layout for data ingestion
with st.sidebar:
    st.markdown("### Data Ingestion")
    st.caption("Paste source text here to refresh the cloud index.")

    user_text = st.text_area(
        "Source Text",
        height=240,
        placeholder="Paste text, notes, docs, or FAQs here...",
        label_visibility="collapsed",
    )
    process_button = st.button("Upload to Cloud DB", type="primary", use_container_width=True)

    st.markdown(
        "<div class='hint'>Tip: shorter chunks usually create faster, more precise retrieval.</div>",
        unsafe_allow_html=True,
    )

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

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown("#### Chat")
st.caption("Ask about the text you uploaded and get grounded answers from the cloud-backed knowledge base.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about your uploaded data..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Querying Cloud Systems..."):
            try:
                answer = query_rag_system(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error querying cloud backend: {e}")

st.markdown("</div>", unsafe_allow_html=True)