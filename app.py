import streamlit as st
import os

# Hydrate environment from Streamlit secrets first so backend init can see keys.
for _secret_key in ("GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"):
    if not os.getenv(_secret_key):
        secret_value = st.secrets.get(_secret_key)
        if secret_value:
            os.environ[_secret_key] = str(secret_value)

import backend as bg  

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
        .stApp, .stApp p, .stApp li, .stApp label,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp [data-testid="stMarkdownContainer"] {
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
            color: #0f172a !important;
        }
        .stChatInput textarea {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.95);
            color: #0f172a !important;
        }
        .stChatInput textarea::placeholder,
        .stTextArea textarea::placeholder {
            color: #64748b !important;
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
        <div class="eyebrow">Enterprise-grade RAG workspace</div>
        <h1>Multimodal RAG Engine with Visual OCR</h1>
        <p>Index documents using sliding window matrices, transcribe handwritten notes on the fly, and run contextually grounded semantic Q&A queries.</p>
        <div class="stat-grid">
            <div class="stat-card"><span>Deployment style</span><strong>Streamlit Front End</strong></div>
            <div class="stat-card"><span>Storage model</span><strong>Pinecone Serverless Vector Store</strong></div>
            <div class="stat-card"><span>Inference path</span><strong>Gemini Multimodal</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard rails for checking secrets
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please verify your Secrets are configured on your Streamlit Cloud dashboard.")
    st.stop()

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    stats = bg.get_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    st.metric(label="Live Index Vector Count", value=f"{total_vectors} Chunks")
    st.divider()

    st.markdown("### Data Ingestion Central")
    
    # UPGRADE FEATURE: Ingestion Channel Separation via Tabs
    tab_text, tab_ocr = st.tabs(["✍️ Plain Text", "📷 Image OCR"])
    
    final_text_to_upload = ""
    
    with tab_text:
        user_text = st.text_area(
            "Source Text Input",
            height=150,
            placeholder="Paste technical parameters, code documentation, or meeting logs...",
            key="plain_text_area"
        )
        if user_text:
            final_text_to_upload = user_text

    with tab_ocr:
        st.caption("Upload images of sketches, chalkboards, or journals.")
        
        # Ingestion Path A: Traditional file picker selection
        uploaded_image = st.file_uploader("Choose Note Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        # Ingestion Path B: Mobile/Desktop live snapshot access
        camera_image = st.camera_input("Or click a photo of your notes:")
        
        target_file = uploaded_image if uploaded_image is not None else camera_image
        
        if target_file:
            # Determine correct document classification mapping
            file_extension = target_file.name.split(".")[-1].lower() if hasattr(target_file, "name") else "jpeg"
            mtype = f"image/{file_extension}" if file_extension in ["png", "jpeg", "jpg"] else "image/jpeg"
            
            with st.spinner("Extracting handwritten elements..."):
                raw_bytes = target_file.read()
                ocr_transcription = bg.process_image_ocr(raw_bytes, mime_type=mtype)
                
                if ocr_transcription:
                    st.success("OCR Extraction Completed!")
                    with st.expander("📝 View Extracted Transcript"):
                        st.write(ocr_transcription)
                    final_text_to_upload = ocr_transcription
                else:
                    st.error("OCR Pipeline failed to decode visual elements.")
    
    st.divider()
    st.markdown("**Advanced Slicing Matrix:**")
    c_size = st.slider("Chunk Size (Chars)", min_value=100, max_value=1200, value=600, step=50)
    c_overlap = st.slider("Overlap Buffer", min_value=0, max_value=300, value=100, step=10)

    process_button = st.button("Upload Assets to Vector Cloud", type="primary", use_container_width=True)

    if process_button:
        if final_text_to_upload.strip():
            with st.spinner("Uploading chunks to Pinecone..."):
                if bg.upload_text_to_cloud(final_text_to_upload, c_size, c_overlap):
                    st.success("Successfully indexed inside Pinecone Cloud!")
                    st.date_input
                    st.rerun()
                else:
                    st.error("Failed to upload. Verify connectivity or index configuration shape.")
        else:
            st.warning("No context loaded. Input text or provide a valid handwritten image resource.")
            
    st.divider()
    st.markdown("### Admin Utilities")
    if st.button("Wipe Vector Database", use_container_width=True):
        with st.spinner("Clearing remote vectors..."):
            if bg.clear_vector_database():
                st.success("Database successfully wiped clean!")
                st.rerun()
            else:
                st.error("Failed to wipe index.")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown("#### Chat Workspace")
st.caption("Ask about the text or hand-written notes you uploaded.")

# Render persistent messages with custom source expanders
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔍 Retained Source Receipts"):
                for source in message["sources"]:
                    st.info(source)

if user_query := st.chat_input("Ask something about your uploaded data..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Querying Cloud Systems..."):
            answer, sources = bg.query_rag_system(user_query)
            st.markdown(answer)
            
            if sources:
                with st.expander("🔍 Retained Source Receipts"):
                    for source in sources:
                        st.info(source)
                        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })

st.markdown("</div>", unsafe_allow_html=True)