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

QuickRag is a simple and minimal package to create **RAG workflows** from PDFs.  
It allows you to load PDF documents, split them into chunks, and store them in a **ChromaDB vector store** for retrieval-augmented generation tasks.

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

