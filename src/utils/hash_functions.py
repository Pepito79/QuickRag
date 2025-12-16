import hashlib
from langchain_core.documents import Document
from datamodel.CoolChunk import CoolChunk


def generate_ids(docs: list[Document] | list[CoolChunk]) -> list[int]:
    """Generate unique id for each document

    Args:
        docs (list[Document | list [CoolChunk]): documents

    Returns:
        list[int]: list of ids
    """

    if isinstance(docs[0], CoolChunk):
        return [
            hashlib.md5(doc.contextualized_chunk.encode()).hexdigest() for doc in docs
        ]
    else:
        return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]
