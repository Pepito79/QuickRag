from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document


def load_pdf_docs(path: str) -> list[Document]:
    """Load pdf documents

    Raises:
        ValueError: if documents not loaded
    Returns:
        list[Document]: chunks returned
    """

    loader = DirectoryLoader(
        path=path,
        glob="**/*.pdf",
        use_multithreading=True,
        recursive=True,
        show_progress=False,
    )

    docs = loader.load()

    if not docs:
        raise ValueError("The loader failed to load your documents")

    else:
        print("Documents loaded successfully")
        return docs
