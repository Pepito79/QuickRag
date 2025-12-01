from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from ragatouille import RAGPretrainedModel


class Rerankers(ABC):

    @abstractmethod
    def rank_docs(self, query: str, docs: list[str]) -> list[str]:
        """Rank the documents

        Args:
            query (str): the user query
            docs (list[str]): the docs to ranks

        Returns:
            list[str]: the content of the document ranked
        """

        pass


class Reranker(Rerankers):
    """Class of a classic reranker with a crossEncoder model"""

    def __init__(
        self,
        top_k: int = 5,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):

        self.cross_encoder = CrossEncoder(model)
        self.top_k = top_k

    def rank_docs(
        self,
        query: str,
        docs: list[str],
    ) -> list[str]:

        # Verify that the user gives a query and list of docs
        if query is None:
            raise ValueError("No query has been given")
        if docs is None:
            raise ValueError("No documents have been given")
        if len(docs) == 0:
            raise ValueError("The list of documents is empty ")

        ranked_text = []
        try:
            # Rank the docs
            ranked_docs = self.cross_encoder.rank(query, docs, self.top_k)
            # Extract the ranked text because .rank outputs a list of dict
            ranked_text = [docs[v["corpus_id"]] for v in ranked_docs]
        except Exception as e:
            print("Exception raised while trying to rank the documents :\n", e)
        return ranked_text


class ColbertReranker(Rerankers):
    """Class that uses a Colbert model to rerank documents"""

    def __init__(self, top_k: int = 5, colbertModel: str = "colbert-ir/colbertv2.0"):

        self.top_k = top_k
        self.colbertModel = RAGPretrainedModel.from_pretrained(colbertModel, -1, 0)

    def rank_docs(self, query, docs):

        # Verify that the user gives a query and list of docs
        if not query:
            raise ValueError("No query has been given")
        if not docs:
            raise ValueError("No query has been given")

        ranked_text = []
        try:
            ranked_docs = self.colbertModel.rerank(
                query=query, documents=docs, k=self.top_k, zero_index_ranks=True
            )
            ranked_text = [doc["content"] for doc in ranked_docs]
            return ranked_text
        except Exception as e:
            print("Exception raised while trying to rank the documents :\n", e)
