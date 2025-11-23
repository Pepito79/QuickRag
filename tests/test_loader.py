from QuickRag.QuickRag import QuickRag
import pytest
from langchain_core.documents import Document


@pytest.fixture(scope="module")
def rag_and_docs():
    doc_path = "/home/pepito/Documents/Python/ML/GenAI/RAG/pdf_documents"
    rag = QuickRag()
    docs = rag.load_pdf_docs(path=doc_path)
    return rag, docs
    

def test_chunk_docs(rag_and_docs):
    rag,docs = rag_and_docs
    chunks = rag.split_docs(docs,512)
    
    assert chunks is not None,"No chunks have been generated"
    assert isinstance(chunks,list]) , "The type of chunks is not a list of Document"
    assert len(chunks) > 0 