import hashlib
from langchain_core.documents import Document


def generate_ids(docs: list[Document]) -> list[int]:
    """Generate unique id for each document

    Args:
        docs (list[Document): documents

    Returns:
        list[int]: list of ids
    """
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]