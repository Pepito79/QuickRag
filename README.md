# QuickRag


<p align="center">
    <img src="assets/qr.png" width="120" alt="QuickRag Logo">
    <br>
    <img src="https://img.shields.io/badge/🐍Python-3.12-00d9ff?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
    <a href="https://github.com/ton_compte/QuickRag">
        <img src="https://img.shields.io/badge/QuickRag-Ready-ff6b6b?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
    </a>
    <img src="https://img.shields.io/badge/License-MIT-4ecdc4?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

# QuickRag - Project Overview

QuickRag is an open-source toolkit for **Retrieval-Augmented Generation (RAG)**.  
It combines document indexing, retrieval, optional reranking, and querying large language models (LLMs) like Gemini.

---

## 🚀 Quick Start

### Clone and install
```bash
git clone https://github.com/ton_compte/QuickRag.git
cd QuickRag

# optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.venv\Scripts\activate     # Windows

# install dependencies
pip install -r requirements.txt
pip install -e .            # editable install for development
```

---

## 🏗️ Architecture Overview

The project is structured around **four main layers**:


---

## 🚀 Core Components

### 1. **ColbertRAG**
Handles **document indexing and retrieval** with ColBERT.

Key features:

- `create_index_from_docs(doc_path, index_name)`: index documents from PDFs.
- `retrieve_docs(query, index_path, top_k)`: fetch top-k documents.
- `query(query, index_path, reranker=None, top_k=5)`: retrieves documents, optionally reranks, and queries Gemini.

Example usage:

```python
colbert = ColbertRAG()
index_path = colbert.create_index_from_docs("docs/", index_name="my_index")
retrieved_docs = colbert.retrieve_docs("What is neural search?", index_path)
```
### 2. Rerankers

Abstract class for document reranking:

```python
from Reranker import Reranker
reranker = Reranker(top_k=5)
ranked_docs = reranker.rank_docs(query="Neural search?", docs=retrieved_docs)
```

*Built-in implementations:*
- Reranker: CrossEncoder-based ranking.
- ColbertReranker: Uses ColBERT for reranking.

### 3. VectorStore

Persistent storage for embeddings using **ChromaDB**.
Supports multiple collections and provides methods to:

- `create_collection(name, embedding_model)`: create or retrieve a collection.

- `add_doc_to_collection(docs, collection)`: add new documents.

- `retrieve_docs(query, collection, top_k)`: retrieve top-k relevant documents.

- `delete_collection(name)` and `list_collection()`.

```python 
db = VectorStore(db_name="my_db")
collection = db.create_collection("my_collection", embedding_model_name="intfloat/multilingual-e5-small")
db.add_doc_to_collection(chunks, collection)
docs = db.retrieve_docs("What is neural search?", collection)
```

### 4.**QuickRag**

High-level wrapper to create a full RAG pipeline:

- Loads and chunks documents.
- Stores embeddings in a vector store.
- Retrieves top-k documents.
- Optionally reranks documents.
- Queries Gemini LLM for answers

**Example:**
```python
qr = QuickRag()
answer = qr.create_naive_gemini(
    path_documents="docs/",
    query="What is neural search?",
    gemini_model="gemini-2.5-flash",
    reranker=reranker
)
print(answer)
```