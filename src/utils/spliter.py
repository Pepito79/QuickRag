from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
import time


def split_docs(
    tokenizer_model: str,
    docs: list[Document],
    chunk_size: int = 512,
) -> list[Document]:
    """Split the documents into chunks, with a recursivce splitter and
        a tokenizer for counting the length .

        Args:
            docs (list[Document]): Docs to chunk
            chunk_size (int, optional): Size of the chunks. Defaults to 512.

    Returns:
            list[Document]: List of chunks where every chunk is a document
    """

    if docs is None:
        raise ValueError("No documents have been provided")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Use the tokenizer from hugging face to count length
    textsplitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        separators=["\n\n", "\n", " ", "", "\n\n"],
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    if textsplitter is None:
        raise ValueError("Failed to create the text splitter")

    # Chunks the document and times it
    print("Chunking started ..... ")
    start = time.time()
    chunks = textsplitter.split_documents(docs)
    end = time.time()
    print(f"Chunking took  {end - start:.4f} seconds ")

    if chunks is None:
        raise ValueError("Documents have not been chunked ")

    return chunks
