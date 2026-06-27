import streamlit as st
import backend as bg

st.set_page_config(
    page_title="RAG Portfolio",
    layout="wide"
)

# ==========================
# SESSION STATE
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ==========================
# UI STYLING
# ==========================

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f8fafc;
    color: #111827;
}

/* Fix ALL text inside chat */
.stChatMessage,
.stChatMessage div,
.stChatMessage span,
.stChatMessage p,
.stChatMessage li {
    color: #111827 !important;
}

/* User + assistant bubbles */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
}

/* Force markdown inside chat to be visible */
[data-testid="stChatMessage"] * {
    color: #111827 !important;
}

/* Title fix */
h1, h2, h3, h4 {
    color: #111827 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* Expander text (sources) */
.streamlit-expanderHeader {
    color: #111827 !important;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* Make ALL sidebar text white */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Optional: improve input visibility */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: #ffffff !important;
    background-color: #1e293b;
}
div.stButton > button {
    background-color: white;
    color: black;
    border: 1px solid #374151;
    border-radius: 10px;
}

div.stButton > button:hover {
    background-color: black !important;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("Atlas Workspace")
st.caption("Search across your uploaded knowledge")

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.header("Library")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    uploaded_img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    # PDF upload
    if uploaded_pdf:
        if uploaded_pdf.name not in st.session_state.indexed_files:
            with st.spinner("Indexing PDF..."):
                text = bg.extract_text_from_pdf(uploaded_pdf)
                count = bg.upload_document(text, uploaded_pdf.name)

            st.session_state.indexed_files.add(uploaded_pdf.name)
            st.success(f"{count} sections indexed")

    # Image upload
    if uploaded_img:
        if uploaded_img.name not in st.session_state.indexed_files:
            with st.spinner("Running OCR..."):
                text = bg.process_image_ocr(uploaded_img.read())
                count = bg.upload_document(text, uploaded_img.name)

            st.session_state.indexed_files.add(uploaded_img.name)
            st.success(f"{count} OCR sections indexed")

# ==========================
# CHAT HISTORY
# ==========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==========================
# QUERY INPUT
# ==========================

query = st.chat_input("Search notes, files, or ask a question...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, sources = bg.generate_answer(query)

        st.write(answer)

        # Sources UI
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"""
                    <div class="source-box">
                        <b>{src.get('source', 'unknown')}</b><br><br>
                        {src.get('text', '')[:300]}...
                    </div>
                    """, unsafe_allow_html=True)

        # Followups
        followups = bg.generate_followups(query, answer)

        if followups:
            st.markdown("### Explore next")

            for q in followups:
                if st.button(q, key=q):

                    # prevent duplicate insertion
                    if q not in [m["content"] for m in st.session_state.messages]:

                        st.session_state.messages.append({
                            "role": "user",
                            "content": q
                        })

                        st.rerun()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
