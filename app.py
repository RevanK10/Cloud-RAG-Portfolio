import streamlit as st
import backend as bg

st.set_page_config(
    page_title="RAG Portfolio",
    layout="wide"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
    color: #111827;
}

.stChatMessage,
.stChatMessage div,
.stChatMessage span,
.stChatMessage p,
.stChatMessage li {
    color: #111827;
}

[data-testid="stChatMessage"] {
    background-color: transparent;
}

[data-testid="stChatMessage"] * {
    color: #111827;
}

h1, h2, h3, h4 {
    color: #111827;
}

section[data-testid="stSidebar"] * {
    color: #111827;
}

.streamlit-expanderHeader {
    color: #111827;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: #ffffff;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: #ffffff;
    background-color: #1e293b;
}
div.stButton > button {
    background-color: white;
    color: black;
    border: 1px solid #374151;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Atlas Workspace")
st.caption("Search across your uploaded knowledge")

with st.sidebar:
    st.header("Library")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    uploaded_img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_pdf:
        if uploaded_pdf.name not in st.session_state.indexed_files:
            with st.spinner("Indexing PDF..."):
                text = bg.extract_text_from_pdf(uploaded_pdf)
                count = bg.upload_document(text, uploaded_pdf.name)

            st.session_state.indexed_files.add(uploaded_pdf.name)
            st.success(f"{count} sections indexed")

    if uploaded_img:
        if uploaded_img.name not in st.session_state.indexed_files:
            with st.spinner("Running OCR..."):
                text = bg.process_image_ocr(uploaded_img.read())
                count = bg.upload_document(text, uploaded_img.name)

            st.session_state.indexed_files.add(uploaded_img.name)
            st.success(f"{count} OCR sections indexed")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


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

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"""
                    <div class="source-box">
                        <b>{src.get('source', 'unknown')}</b><br><br>
                        {src.get('text', '')[:300]}...
                    </div>
                    """, unsafe_allow_html=True)

        followups = bg.generate_followups(query, answer)

        if followups:
            st.markdown("### Explore next")

            for q in followups:
                if st.button(q, key=q):

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
