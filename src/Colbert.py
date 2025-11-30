from ragatouille import RAGPretrainedModel
import os
from utils.loader import load_pdf_docs
from utils.spliter import split_docs
from utils.hash_functions import generate_ids
from utils.api_key_verifier import verify_gemini_key
from utils.gemini import get_answer_gemini


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

    def query(
        self,
        query: str,
        index_path: str,
        gemini_model: str = "gemini-2.5-flash",
    ):

        # Verify if the user has a gemini key in his env
        verify_gemini_key()

        # Load the retriever and use it as a langchain retriever
        retriever = self.colbert_model.from_index(
            index_path=index_path, verbose=1
        ).as_langchain_retriever(k=5)

        # Get the retrieved docs
        retrieved_docs = retriever.invoke(query)

        answer = get_answer_gemini(model=gemini_model, query=query, docs=retrieved_docs)

        print(answer)
        return answer
