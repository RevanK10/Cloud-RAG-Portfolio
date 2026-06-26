import streamlit as st
import os
import time
import json
from datetime import datetime

# Hydrate environment from Streamlit secrets first.
for _secret_key in ("GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"):
    if not os.getenv(_secret_key):
        secret_value = st.secrets.get(_secret_key)
        if secret_value:
            os.environ[_secret_key] = str(secret_value)

import backend as bg

st.set_page_config(
    page_title="Enterprise RAG Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(37,99,235,0.13) 0%, transparent 40%),
        radial-gradient(circle at 92% 4%, rgba(16,185,129,0.11) 0%, transparent 34%),
        linear-gradient(175deg, #f0f6ff 0%, #eaf1ff 55%, #f5f9ff 100%);
    color: #0f172a;
}
.stApp, .stApp p, .stApp li, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] { color: #0f172a; }
.block-container { padding-top: 1.75rem; padding-bottom: 2rem; }

/* Hero */
.hero {
    padding: 2.25rem 2.5rem;
    border-radius: 28px;
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(148,163,184,0.2);
    box-shadow: 0 20px 56px rgba(15,23,42,0.07), 0 1px 2px rgba(15,23,42,0.04);
    backdrop-filter: blur(20px);
    margin-bottom: 1.5rem;
}
.eyebrow {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.78rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; color: #2563eb; margin-bottom: 0.75rem;
}
.hero h1 {
    font-size: clamp(2rem, 3.5vw, 3.6rem);
    line-height: 1.04; margin: 0; color: #0f172a;
    font-weight: 800; letter-spacing: -0.02em;
}
.hero p { margin: 0.85rem 0 0; max-width: 64ch; color: #475569; font-size: 1.05rem; line-height: 1.65; }
.stat-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem; margin-top: 1.6rem;
}
.stat-card {
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(241,247,255,0.9));
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 4px 16px rgba(15,23,42,0.05);
    padding: 1rem 1.2rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(37,99,235,0.10); }
.stat-card span { display: block; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em; color: #64748b; margin-bottom: 0.4rem; font-weight: 600; }
.stat-card strong { display: block; font-size: 0.95rem; color: #0f172a; font-weight: 700; }

/* Panel */
.panel-card {
    border-radius: 24px; background: rgba(255,255,255,0.84);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 12px 40px rgba(15,23,42,0.06);
    padding: 1.5rem 1.75rem; backdrop-filter: blur(16px);
}

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0c1425 0%, #111827 100%); }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #e2eaf8 !important; }
[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stTabs [aria-selected="true"] { color: #60a5fa !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%; border-radius: 14px; border: 0;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white; font-weight: 700; padding: 0.8rem 1rem;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35);
    transition: opacity 0.15s ease, transform 0.1s ease;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.92; transform: translateY(-1px); }
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.07) !important; color: #e2eaf8 !important;
    border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 12px;
}
[data-testid="stMetric"] { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 0.75rem; }

/* Chat */
.stChatMessage { border-radius: 16px; padding: 0.2rem 0; }
[data-testid="stChatMessageContent"] p { color: #0f172a !important; }
.stChatInput textarea, .stTextArea textarea {
    border-radius: 16px !important; border: 1.5px solid rgba(148,163,184,0.3) !important;
    background: rgba(255,255,255,0.95) !important; color: #0f172a !important; font-size: 0.97rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stChatInput textarea:focus, .stTextArea textarea:focus {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* Citations */
.citation-box {
    background: linear-gradient(135deg, #f0fdf9, #ecfdf5);
    border-left: 4px solid #10b981; padding: 0.8rem 1.1rem; border-radius: 10px;
    margin-bottom: 0.65rem; box-shadow: 0 2px 10px rgba(16,185,129,0.08);
}
.cite-meta { font-size: 0.82rem; color: #065f46; font-weight: 600; margin-bottom: 0.3rem; }
.cite-text { color: #374151; font-size: 0.9rem; line-height: 1.55; font-style: italic; }

/* Correction banner */
.correction-banner {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 1px solid #f59e0b; border-radius: 12px; padding: 0.6rem 1rem;
    font-size: 0.87rem; color: #78350f; font-weight: 600; margin-top: 0.5rem;
}

/* Follow-up pills */
.followup-container { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }
.followup-pill {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe; border-radius: 999px;
    padding: 0.38rem 0.9rem; font-size: 0.83rem; color: #1d4ed8;
    cursor: pointer; font-weight: 500;
    transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.1s ease;
    display: inline-block;
}
.followup-pill:hover { background: #dbeafe; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(37,99,235,0.15); }

/* Meta timing badge */
.timing-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.75rem; color: #64748b; background: rgba(100,116,139,0.08);
    border-radius: 999px; padding: 0.2rem 0.65rem; margin-top: 0.5rem;
}

/* Knowledge browser */
.kb-row {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(255,255,255,0.06); border-radius: 10px;
    padding: 0.6rem 0.9rem; margin-bottom: 0.4rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.kb-name { color: #e2eaf8; font-size: 0.88rem; font-weight: 500; }
.kb-meta { color: #64748b; font-size: 0.78rem; }

/* Progress bar override */
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #10b981); border-radius: 999px; }

/* Copy button */
.copy-btn {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px; color: #94a3b8; font-size: 0.78rem; padding: 0.2rem 0.6rem;
    cursor: pointer; transition: background 0.15s;
}
.copy-btn:hover { background: rgba(255,255,255,0.15); color: #e2eaf8; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="eyebrow">⚡ Enterprise Hybrid Matrix v3.0</div>
    <h1>Self-Correcting Multi-Agent RAG</h1>
    <p>Index multi-format knowledge assets, extract layouts via high-density visual OCR, and conduct
    conversations protected by real-time agentic faithfulness guardrails — powered by Gemini 2.5.</p>
    <div class="stat-grid">
        <div class="stat-card"><span>Architecture</span><strong>🛡️ Self-Correcting Guardrails</strong></div>
        <div class="stat-card"><span>Retrieval</span><strong>🔁 Query Rewrite + Re-rank</strong></div>
        <div class="stat-card"><span>Vector Infrastructure</span><strong>📌 Pinecone Batch Upsert</strong></div>
        <div class="stat-card"><span>Intelligence Engine</span><strong>✨ Gemini 2.5 Real-Time</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("⚠️ Missing API Keys — configure **GOOGLE_API_KEY** and **PINECONE_API_KEY** in your deployment secrets.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "staged_text": "",
    "staged_filename": "Manual Entry",
    "follow_ups": [],
    "pending_query": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Handle follow-up pill clicks (injected as pending query) ──────────────────
if st.session_state.pending_query:
    pending = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": pending})
    # Will be processed below in the normal chat flow
    st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    stats = bg.get_index_stats()
    total_vectors = stats.get("total_vector_count", 0)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Knowledge Chunks", f"{total_vectors:,}")
    with col2:
        st.metric("Messages", f"{len(st.session_state.messages):,}")
    st.divider()

    # ── Ingestion tabs ────────────────────────────────────────────────────────
    st.markdown("### 📥 Ingestion Dashboard")
    tab_doc, tab_text, tab_ocr = st.tabs(["📄 PDF", "✍️ Text", "📷 OCR"])

    with tab_doc:
        uploaded_pdf = st.file_uploader("Drop an enterprise PDF", type=["pdf"], key="pdf_uploader")
        if uploaded_pdf:
            with st.spinner("Extracting document layers…"):
                extracted = bg.extract_text_from_pdf(uploaded_pdf)
                if extracted:
                    st.session_state.staged_text = extracted
                    st.session_state.staged_filename = uploaded_pdf.name
                    st.success(f"Extracted {len(extracted):,} characters.")
                else:
                    st.error("Could not extract text from this PDF.")

    with tab_text:
        user_text = st.text_area(
            "Source Text", height=150,
            placeholder="Paste logs, technical notes, or instructions…",
            key="plain_text_area",
        )
        if user_text:
            st.session_state.staged_text = user_text
            st.session_state.staged_filename = "Manual Text Entry"

    with tab_ocr:
        st.caption("Extract text from sketches, forms, or photos.")
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="img_uploader")
        camera_image = st.camera_input("Or capture a live snapshot:")
        target_file = uploaded_image if uploaded_image is not None else camera_image
        if target_file:
            ext = target_file.name.split(".")[-1].lower() if hasattr(target_file, "name") else "jpeg"
            mtype = f"image/{ext}" if ext in ["png", "jpeg", "jpg"] else "image/jpeg"
            with st.spinner("Running Gemini OCR…"):
                raw_bytes = target_file.read()
                ocr_result = bg.process_image_ocr(raw_bytes, mime_type=mtype)
            if ocr_result:
                st.success("OCR extraction complete!")
                with st.expander("🔍 View OCR Transcript"):
                    st.write(ocr_result)
                st.session_state.staged_text = ocr_result
                st.session_state.staged_filename = f"OCR ({ext})"
            else:
                st.error("OCR returned no content.")

    st.divider()

    if st.session_state.staged_text:
        char_count = len(st.session_state.staged_text)
        st.info(f"📋 **Staged:** {st.session_state.staged_filename}  \n`{char_count:,}` characters ready to index")

    st.markdown("**Chunking Configuration**")
    c_size = st.slider("Chunk Length", min_value=100, max_value=1200, value=600, step=50)
    c_overlap = st.slider("Sliding Overlap", min_value=0, max_value=300, value=100, step=10)

    if st.button("⬆️ Index Asset into Vector Store", type="primary", use_container_width=True):
        if st.session_state.staged_text.strip():
            progress_bar = st.progress(0, text="Preparing chunks…")
            status_text = st.empty()

            def on_progress(done, total):
                pct = done / total
                progress_bar.progress(pct, text=f"Embedding chunk {done}/{total}…")
                status_text.caption(f"`{done}/{total}` chunks vectorised")

            chunk_count = bg.upload_text_to_cloud(
                st.session_state.staged_text, c_size, c_overlap,
                source_file=st.session_state.staged_filename,
                progress_callback=on_progress,
            )
            progress_bar.empty()
            status_text.empty()

            if chunk_count > 0:
                st.success(f"✅ {chunk_count} chunks indexed successfully!")
                st.session_state.staged_text = ""
                st.session_state.staged_filename = "Manual Entry"
                st.rerun()
            else:
                st.error("Upsert failed — check your Pinecone configuration.")
        else:
            st.warning("No staged content to index. Add a document first.")

    st.divider()

    # ── Knowledge Base Browser ────────────────────────────────────────────────
    st.markdown("### 🗂️ Knowledge Base")
    if st.button("🔄 Refresh Index", use_container_width=True):
        st.rerun()

    sources = bg.get_indexed_sources()
    if sources:
        for src in sources:
            uploaded_dt = datetime.fromtimestamp(src["uploaded"]).strftime("%b %d, %H:%M")
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(
                    f'<div class="kb-row">'
                    f'<div>'
                    f'<div class="kb-name">📄 {src["source"][:28]}{"…" if len(src["source"]) > 28 else ""}</div>'
                    f'<div class="kb-meta">{src["chunks"]} chunks · {uploaded_dt}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{src['source']}", help=f"Delete {src['source']}"):
                    with st.spinner(f"Deleting {src['source']}…"):
                        bg.delete_source(src["source"])
                    st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # ── Cluster utilities ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Utilities")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Wipe Index", use_container_width=True):
            with st.spinner("Wiping…"):
                if bg.clear_vector_database():
                    st.session_state.messages = []
                    st.session_state.staged_text = ""
                    st.session_state.follow_ups = []
                    st.success("Wiped.")
                    st.rerun()
                else:
                    st.error("Failed.")
    with col_b:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.follow_ups = []
            st.rerun()

    # ── Export conversation ───────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        md_lines = [f"# RAG Conversation Export\n*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]
        for msg in st.session_state.messages:
            role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
            md_lines.append(f"{role_label}\n\n{msg['content']}\n")
            if msg.get("sources"):
                md_lines.append("*Sources: " + ", ".join(set(s["file"] for s in msg["sources"])) + "*\n")
        export_md = "\n---\n".join(md_lines)
        st.download_button(
            "📥 Export Conversation (.md)",
            data=export_md,
            file_name=f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ── MAIN CHAT ─────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown("#### Operational Intelligence Workspace")
st.caption("Ask anything about your indexed documents. The engine rewrites vague queries, re-ranks results, and self-corrects hallucinations automatically.")


for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("latency_ms"):
                st.markdown(
                    f'<div class="timing-badge">⏱ {message["latency_ms"]}ms response</div>',
                    unsafe_allow_html=True,
                )

            # Correction banner
            if message.get("corrected"):
                st.markdown(
                    '<div class="correction-banner">🛡️ Faithfulness guardrail applied — output corrected for context accuracy.</div>',
                    unsafe_allow_html=True,
                )

            # Citations
            if message.get("sources"):
                with st.expander("🔗 Source Citations & Confidence Scores"):
                    for src in message["sources"]:
                        chunk_info = f" · chunk {src['chunk']}" if src.get("chunk") and src["chunk"] != "—" else ""
                        st.markdown(
                            f"""<div class="citation-box">
                                <div class="cite-meta">📄 {src['file']}{chunk_info} &nbsp;·&nbsp; 🎯 {src['confidence']} match</div>
                                <div class="cite-text">{src['text'][:300]}{'…' if len(src['text']) > 300 else ''}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # Follow-up pills (only on the last assistant message)
            if i == len(st.session_state.messages) - 1 and message.get("follow_ups"):
                st.markdown("**Suggested follow-ups:**")
                cols = st.columns(len(message["follow_ups"]))
                for j, q in enumerate(message["follow_ups"]):
                    with cols[j]:
                        if st.button(q, key=f"followup_{i}_{j}", use_container_width=True):
                            st.session_state.pending_query = q
                            st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if user_query := st.chat_input("Ask a question about your documents…"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        t_start = time.time()

        # 1. Retrieve with query rewriting + re-ranking
        prior_history = st.session_state.messages[:-1]
        with st.spinner("Searching & re-ranking knowledge index…"):
            context, sources = bg.query_rag_system_context(user_query, history=prior_history)

        if context:
            # 2. Stream primary answer
            stream_gen = bg.generate_streaming_chat_response(user_query, context, prior_history)
            raw_answer = st.write_stream(stream_gen)

            # 3. Faithfulness self-correction
            corrected = False
            with st.spinner("Running faithfulness verification…"):
                final_answer = bg.self_correct_response(user_query, raw_answer, context)

            if final_answer.strip() and final_answer.strip() != raw_answer.strip():
                corrected = True
                st.empty()
                st.markdown(final_answer)
                st.markdown(
                    '<div class="correction-banner">🛡️ Faithfulness guardrail applied — output corrected for context accuracy.</div>',
                    unsafe_allow_html=True,
                )
                raw_answer = final_answer

            # 4. Latency badge
            latency_ms = int((time.time() - t_start) * 1000)
            st.markdown(
                f'<div class="timing-badge">⏱ {latency_ms:,}ms response</div>',
                unsafe_allow_html=True,
            )

            # 5. Citations
            if sources:
                with st.expander("🔗 Source Citations & Confidence Scores"):
                    for src in sources:
                        chunk_info = f" · chunk {src['chunk']}" if src.get("chunk") and src["chunk"] != "—" else ""
                        st.markdown(
                            f"""<div class="citation-box">
                                <div class="cite-meta">📄 {src['file']}{chunk_info} &nbsp;·&nbsp; 🎯 {src['confidence']} match</div>
                                <div class="cite-text">{src['text'][:300]}{'…' if len(src['text']) > 300 else ''}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # 6. Generate follow-up questions
            with st.spinner("Generating follow-up suggestions…"):
                follow_ups = bg.generate_follow_up_questions(user_query, raw_answer, context)

            if follow_ups:
                st.markdown("**Suggested follow-ups:**")
                cols = st.columns(len(follow_ups))
                for j, q in enumerate(follow_ups):
                    with cols[j]:
                        if st.button(q, key=f"new_followup_{j}", use_container_width=True):
                            st.session_state.pending_query = q
                            st.rerun()

        else:
            raw_answer = "I cannot find relevant information in the indexed documents for that query. Try uploading related source material first."
            corrected = False
            latency_ms = int((time.time() - t_start) * 1000)
            follow_ups = []
            st.markdown(raw_answer)

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_answer,
        "sources": sources if context else [],
        "corrected": corrected,
        "latency_ms": latency_ms if context else None,
        "follow_ups": follow_ups if context else [],
    })

st.markdown("</div>", unsafe_allow_html=True)