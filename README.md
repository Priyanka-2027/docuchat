# 📄 DocuChat

A RAG (Retrieval-Augmented Generation) powered PDF chatbot that lets you upload any PDF and ask questions about it in natural language. Built with LangChain, ChromaDB, HuggingFace embeddings, and Google Gemini.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-docuchat.streamlit.app/)

---

## ✨ Features

- **Upload any PDF** — resumes, research papers, books, contracts, reports
- **Semantic search** — finds relevant chunks using vector similarity, not just keyword matching
- **Multi-turn chat** — remembers conversation context across multiple questions
- **Auto-retry retrieval** — if the first retrieval misses context, automatically retries with broader search
- **Source transparency** — shows exactly which chunks of the document were used to answer
- **Configurable** — adjust chunk size, overlap, and number of retrieved chunks from the sidebar
- **Gemini 2.5 Flash** — fast, accurate answers grounded strictly in the document

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Vector Store | ChromaDB |
| PDF Loading | LangChain + PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| Orchestration | LangChain |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader  ──►  RecursiveCharacterTextSplitter  ──►  HuggingFace Embeddings
                                                              │
                                                              ▼
                                                         ChromaDB (Vector Store)
                                                              │
User Query ──► MMR Similarity Search ──► Retrieved Chunks ──►│
                                                              ▼
                                                    Gemini 2.5 Flash (LLM)
                                                              │
                                                              ▼
                                                         Answer + Sources
```

**RAG Pipeline:**
1. PDF is loaded and split into overlapping chunks
2. Each chunk is embedded using a HuggingFace sentence transformer model
3. Embeddings are stored in ChromaDB (in-memory)
4. On each query, top-k most relevant chunks are retrieved using Max Marginal Relevance (MMR)
5. Retrieved chunks are injected into the LLM prompt as context
6. Gemini answers strictly based on the provided context
7. If the answer indicates missing context, an automatic retry with more chunks is triggered

---

## ⚙️ Setup & Run Locally

### Prerequisites
- Python 3.10+
- A Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/Priyanka-2027/docuchat.git
cd docuchat
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 How to Use

1. Enter your **Gemini API Key** in the sidebar
2. Upload a **PDF file** using the file uploader
3. Click **⚡ Process Document** — this embeds the document (takes ~10-30 seconds)
4. Ask any question in the chat input
5. Expand **📚 Source chunks used** to see what context was retrieved

---

## 🎛️ Configuration Options

| Setting | Default | Description |
|---|---|---|
| Chunk size | 800 | Characters per chunk when splitting the PDF |
| Chunk overlap | 200 | Overlap between consecutive chunks to preserve context |
| Chunks to retrieve (k) | 5 | Number of chunks retrieved per query |

Increase `k` for dense documents where answers span multiple sections.

---

## 📁 Project Structure

```
docuchat/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── docuchat.ipynb          # Original Jupyter notebook (Colab prototype)
└── README.md
```

---

## 🔑 Environment & API Key

The app takes the Gemini API key directly from the sidebar UI — no `.env` file needed. This makes it easy to share and deploy without exposing credentials.

For deployment on Streamlit Cloud, you can optionally set it as a secret:
- Go to your app settings → **Secrets**
- Add: `GEMINI_API_KEY = "your_key_here"`

---

## 🚢 Deployment

### Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `main` branch → `app.py`
5. Click **Deploy**

First load takes ~2 minutes as the embedding model (~420MB) is downloaded and cached.

---

## 📝 Origin

This project started as a Google Colab notebook (`docuchat.ipynb`) exploring the RAG pipeline on the "Attention Is All You Need" paper. It was then converted into a full deployable Streamlit app with PDF upload, configurable retrieval, multi-turn chat, and automatic retry logic.

---

## 📄 License

MIT
