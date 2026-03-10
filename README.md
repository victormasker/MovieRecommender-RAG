# RAG Movie Recommendation System with ObjectBox and LangChain

A Retrieval Augmented Generation (RAG) application that provides intelligent movie recommendations using IMDB review data. Built with LangChain, ObjectBox vector database, and LLaMA3 running locally via Ollama — fully offline, no API keys required.

![Streamlit Web App Interface](./images/RAG%20app%20UI.png)

## How It Works

1. **Load** — IMDB movie reviews are loaded from CSV using LangChain's `CSVLoader`
2. **Chunk** — Text is split into 1000-character chunks (200 overlap) with `RecursiveCharacterTextSplitter`
3. **Embed** — Chunks are converted to 768-dimensional vectors using HuggingFace BGE (`BAAI/bge-small-en-v1.5`)
4. **Store** — Embeddings are stored in ObjectBox, an on-device vector database (no cloud dependency)
5. **Query** — User questions are embedded and matched against stored vectors via similarity search
6. **Generate** — Retrieved context is passed to LLaMA3-8B (via Ollama) to generate answers

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Vector Store | ObjectBox (on-device) |
| Embeddings | HuggingFace BGE (`BAAI/bge-small-en-v1.5`) |
| LLM | LLaMA3-8B via Ollama (local) |
| Web UI | Streamlit |
| Data | IMDB Movie Reviews (CSV) |

## Project Structure

```
├── app/
│   ├── app.py                   # Streamlit web application
│   ├── config.py                # Environment variable management
│   ├── utils.py                 # Embeddings, LLM, and search utilities
│   └── test_recommendations.py  # Standalone recommendation testing script
├── data/
│   └── IMDB Dataset.csv         # Movie reviews dataset
├── objectbox/                   # Pre-built vector database
├── us-census-data/              # Sample PDF documents (for experimentation)
├── requirements.txt
└── .env                         # Environment variables (optional)
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai) installed and running

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/NebeyouMusie/End-to-End-RAG-Project-using-ObjectBox-and-LangChain.git
cd End-to-End-RAG-Project-using-ObjectBox-and-LangChain

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install streamlit sentence-transformers

# 4. Pull the LLaMA3 model via Ollama
ollama pull llama3:8b

# 5. Run the app
cd app
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser (URL shown in terminal)
2. Click **"Embed Documents"** to index the IMDB dataset into ObjectBox (only needed once — pre-built embeddings are included)
3. Type a question or preference in the text field, for example:
   - *"Recommend me some good action movies"*
   - *"What are some highly rated sci-fi films?"*
   - *"I want movies with great humor and positive reviews"*
4. View the generated answer, response time, and similar documents retrieved

## Next Steps: Building a More Advanced RAG System

### Chunking Strategies
- **Semantic chunking** — split text based on meaning rather than fixed character count (e.g., using embedding similarity to find natural breakpoints)
- **Document-aware chunking** — preserve document structure (headers, paragraphs, metadata) during splitting
- **Experiment with chunk sizes** — try smaller chunks (256-512) for precision or larger chunks (1500-2000) for more context

### Better Embeddings
- **Larger embedding models** — try `BAAI/bge-large-en-v1.5` (1024-dim) or `intfloat/e5-large-v2` for higher quality
- **Fine-tune embeddings** — train on your domain-specific data for better retrieval relevance
- **Embedding benchmarks** — evaluate models on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

### Advanced Retrieval
- **Hybrid search** — combine dense vector search with sparse keyword search (BM25) for better recall
- **Re-ranking** — add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder retrieved results
- **Maximal Marginal Relevance (MMR)** — diversify results to avoid returning near-duplicate chunks
- **Multi-vector retrieval** — generate summaries per chunk and retrieve on summaries, then return full chunks

### Query Transformation
- **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer first, then use it to search
- **Multi-query retrieval** — rephrase the user question in multiple ways and merge results
- **Step-back prompting** — ask a more general question first to gather broader context

### Evaluation & Observability
- **RAGAS framework** — measure faithfulness, answer relevance, and context precision
- **LangSmith / LangFuse** — trace and debug your RAG pipeline end-to-end
- **Build an evaluation dataset** — create question-answer pairs from your data to benchmark improvements

### Production Improvements
- **Streaming responses** — show LLM output token-by-token for better UX
- **Conversation memory** — add chat history so users can ask follow-up questions
- **Caching** — cache embeddings and LLM responses to reduce latency
- **Metadata filtering** — filter by genre, year, rating before similarity search

### Multi-Modal RAG
- **Images and tables** — extract and embed non-text content from documents
- **Vision models** — use multi-modal LLMs to reason over images alongside text

### Agentic RAG
- **Tool-using agents** — let the LLM decide when to search, what to search, and how to combine results
- **Self-corrective RAG** — detect low-confidence answers and automatically retry with refined queries
- **Multi-step reasoning** — chain multiple retrieval steps for complex questions

## Acknowledgments

- [Krish Naik](https://www.youtube.com/@krishnaik06) for the original project tutorial
- [NebeyouMusie](https://github.com/NebeyouMusie) for the original codebase
