import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuChat",
    page_icon="📄",
    layout="centered",
)

st.title("📄 DocuChat")
st.caption("Upload a PDF and ask questions about it.")

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None    # ChromaDB instance
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None  # track which file is loaded


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="The document will be split, embedded, and stored locally for retrieval.",
    )

    chunk_size = st.slider("Chunk size", 500, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)

    process_btn = st.button("⚡ Process Document", use_container_width=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Helper: load & index PDF ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def process_pdf(file_bytes: bytes, filename: str, chunk_size: int, chunk_overlap: int):
    """Write PDF to a temp file, load, split, embed, store in Chroma."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name="docuchat_collection",
        embedding_function=embedding_model,
    )
    vector_store.add_documents(documents=splits)

    return vector_store, len(splits), len(docs)


def retrieve_context(vector_store, query: str, k: int):
    # Use MMR (max marginal relevance) to get diverse, relevant chunks
    try:
        retrieved = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
    except Exception:
        retrieved = vector_store.similarity_search(query, k=k)

    context = ""
    for doc in retrieved:
        context += f"Source: {doc.metadata}\nContent: {doc.page_content}\n\n"
    return context, retrieved


# Phrases that indicate the LLM didn't find the answer in context
_NOT_FOUND_PHRASES = [
    "does not contain",
    "not mentioned",
    "no information",
    "cannot find",
    "not provided",
    "not available in",
    "i apologize",
    "i don't have",
    "context does not",
]


def ask_llm(model, context: str, chat_history: list, user_query: str,
            vector_store=None, top_k: int = 5) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided "
        "document context. If the answer is in the context, provide it clearly and "
        "completely. Do not make up information not present in the context.\n\n"
        f"Context:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    response = model.invoke(messages)
    answer = response.content

    # If answer suggests context was missing, retry with more chunks
    if vector_store and any(p in answer.lower() for p in _NOT_FOUND_PHRASES):
        bigger_k = min(top_k * 2, 15)
        retry_docs = vector_store.similarity_search(user_query, k=bigger_k)
        retry_context = ""
        for doc in retry_docs:
            retry_context += f"Source: {doc.metadata}\nContent: {doc.page_content}\n\n"

        retry_system = (
            "You are a helpful assistant. The user asked a question and the first "
            "retrieval attempt missed some context. Here is a broader set of context "
            "from the document. Answer using ONLY this context.\n\n"
            f"Context:\n{retry_context}"
        )
        retry_messages = [{"role": "system", "content": retry_system}]
        for msg in chat_history[-6:]:
            retry_messages.append({"role": msg["role"], "content": msg["content"]})
        retry_messages.append({"role": "user", "content": user_query})

        retry_response = model.invoke(retry_messages)
        return retry_response.content

    return answer


# ── Process document when button clicked ─────────────────────────────────────
if process_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    elif not api_key:
        st.sidebar.error("Please enter your Gemini API key.")
    else:
        with st.spinner("Processing document…"):
            try:
                vs, n_chunks, n_pages = process_pdf(
                    uploaded_file.read(),
                    uploaded_file.name,
                    chunk_size,
                    chunk_overlap,
                )
                st.session_state.vector_store = vs
                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = []  # reset chat for new doc
                st.sidebar.success(
                    f"✅ Done! {n_pages} pages → {n_chunks} chunks indexed."
                )
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {e}")


# ── Show status ───────────────────────────────────────────────────────────────
if st.session_state.processed_file:
    st.info(f"📘 Active document: **{st.session_state.processed_file}**")
else:
    st.warning("No document loaded. Upload a PDF and click **Process Document**.")


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your document…"):
    if not st.session_state.vector_store:
        st.error("Please process a document first.")
    elif not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    else:
        # show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    model = init_chat_model(
                        "google_genai:gemini-2.5-flash",
                        api_key=api_key,
                    )
                    context, source_docs = retrieve_context(
                        st.session_state.vector_store, prompt, top_k
                    )
                    answer = ask_llm(
                        model, context, st.session_state.messages[:-1], prompt,
                        vector_store=st.session_state.vector_store,
                        top_k=top_k,
                    )

                    st.markdown(answer)

                    # optional: show sources in an expander
                    with st.expander("📚 Source chunks used"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Chunk {i}** — page {doc.metadata.get('page', '?')}")
                            st.caption(doc.page_content[:400] + "…")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
