from ragatouille import RAGPretrainedModel
import os
from utils.loader import load_pdf_docs
from utils.spliter import split_docs
from utils.hash_functions import generate_ids
from utils.api_key_verifier import verify_gemini_key
from utils.gemini import get_answer_gemini
from langchain_core.documents import Document
from Reranker import Rerankers


# A util function that allows us to load and split the documents
def _load_and_split(chunk_size, doc_path, tokenizer_name: str):
    docs = load_pdf_docs(doc_path)
    chunks = split_docs(
        tokenizer_model=tokenizer_name, docs=docs, chunk_size=chunk_size
    )
    text_chunks = [c.page_content for c in chunks]
    ids = generate_ids(chunks)
    return text_chunks, ids


class ColbertRAG:

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        tokenizer_name: str = "intfloat/multilingual-e5-small",
    ):

        # Load the colbert model
        self.model_name = model_name
        self.colbert_model = RAGPretrainedModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name, verbose=0
        )
        self.tokenizer_name = tokenizer_name

    def create_index_from_docs(
        self,
        doc_path: str,
        index_name: str,
        chunk_size: int = 512,
    ) -> str:
        """Create an index from a list of documents

        Args:
            doc_path (str): the path of the docs
            index_name (str): the name of the new index
            chunk_size (int, optional): Size of the chunks. Defaults to 512.

        Returns:
            str: the index path
        """

        overwritte = False
        index_path = f"./.ragatouille/colbert/indexes/{index_name}"
        # Verify if the index exists and ask the user if he wants to overwritte it
        if os.path.exists(index_path):
            print(f"Index {index_name} already exists\n")
            ans = input("Do you want to overwrite the actual index ? [y/n]")
            if ans.lower() == "y":
                overwritte = True
            else:
                print("Nothing has been overwritted you index is here !")
                path_to_index = index_path
                return path_to_index

        # Get the chunks and the ids
        text_chunks, ids = _load_and_split(chunk_size, doc_path, self.tokenizer_name)

        # Create the index from the documents
        try:
            index_path = self.colbert_model.index(
                collection=text_chunks,
                document_ids=ids,
                split_documents=False,
                index_name=index_name,
                overwrite_index=overwritte,
            )
            print(f"Index {index_name} successfully created ")
            return index_path

        except Exception as e:
            raise Exception(
                "Exception raised whil trying to create the index from the documents : ",
                e,
            )

    def add_docs(self, docs_path: str, index_name: str):
        """Add documents to an existing index

        Args:
            docs_path (str): the docs path
            index_name (str): the name of the existing index
        """
        text_chunks, ids = _load_and_split(
            chunk_size=512, doc_path=docs_path, tokenizer_name=self.tokenizer_name
        )
        try:
            self.colbert_model.add_to_index(
                index_name=index_name,
                new_collection=text_chunks,
                new_document_ids=ids,
                split_documents=False,
            )
            print(f"Documents added successfully to the index {index_name}")
        except Exception as e:
            raise Exception(
                "Exception raised while trying to add docs to the index\n", e
            )

    def retrieve_docs(
        self, query: str, index_path: str, top_k: int = 5
    ) -> list[Document]:
        """Retreves the documents from an index

        Args:
            query (str): the user query
            index_path (str): the index path
            top_k (int, optional): the top_k documents to keep. Defaults to 5._

        Returns:
            list[Document]: list of documents retrieved
        """

        # Verify if the index path exists
        if not os.path.exists(index_path):
            raise ValueError("The index path does not exists")
        try:
            retriever = self.colbert_model.from_index(
                index_path=index_path, n_gpu=-1, verbose=0
            ).as_langchain_retriever(k=top_k)
        except Exception as e:
            raise Exception("Exception raised while trying to retrieve the documents")

        retrieved_docs = retriever.invoke(query)
        return retrieved_docs

    def query(
        self,
        query: str,
        index_path: str,
        reranker: Rerankers | None = None,
        top_k: int = 5,
        gemini_model: str = "gemini-2.5-flash",
    ) -> str:
        """Query your documents using late interaction with Colbert

        Args:
            query (str): the query
            index_path (str): the path to your exisiting index
            top_k (int, optional): How many documents you want to retrieve. Defaults to 5.
            gemini_model (str, optional): the Gemini model . Defaults to "gemini-2.5-flash".

        Returns:
            str: the answer of the llm
        """
        # Verify if the user has a gemini key in his env
        verify_gemini_key()

        # Use the reranker if it is given
        if reranker is not None:
            retrieved_docs = self.retrieve_docs(query, index_path, top_k)
            ranked_docs = reranker.rank_docs(query=query, docs=retrieved_docs)
            context = ranked_docs

        # If no reranker used
        else:
            # Get the retrieved docs
            context = self.retrieve_docs(query, index_path, top_k)

        answer = get_answer_gemini(model=gemini_model, query=query, docs=context)

        print(answer)
        return answer
