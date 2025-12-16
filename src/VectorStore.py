from langchain_chroma import Chroma
from chromadb import PersistentClient
from chromadb.types import Collection
from langchain_core.documents import Document
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from utils.hash_functions import generate_ids
from datamodel.CoolChunk import CoolChunk
import torch
from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import (
    create_langchain_embedding,
)
import json


class VectorStore:
    """
    VectorStore class that stocks collections and allows to get access / modify / delete
    and add knowledge to it.
    """

    def __init__(
        self,
        db_name: str,
        persistent_dir: str = "./chromDB",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store .

        Args:
            db_name (str): name of the databaser
            persistent_dir (str, optional): Persistent directory where to stock the vector store db.
            Defaults to "./chromDB".
        """

        self.dir = persistent_dir
        self.chromaDB = PersistentClient(path=persistent_dir)
        self.db_name = db_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_function = create_langchain_embedding(
            HuggingFaceEmbeddings(
                model_name=embedding_model, model_kwargs={"device": device}
            )
        )

    def create_collection(
        self,
        collection_name: str,
    ) -> Collection:
        """
        If the collection is not in the db it creates one with an embedding model coming from langchain
        and if it already exists it returns it.

        Returns:
            Collection
        """

        # Create the collection
        try:
            collection = self.chromaDB.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            print("Collection created successfully !")
            return collection

        except Exception as e:
            print("Exception raised\n", e)

    def delete_collection(self, collection_name: str):
        """
        Delete an existing collection from the vector store db

        Args:
            collection_name (str): name of the collection to delete
        """
        self.chromaDB.delete_collection(collection_name)

        print(f"Deleted {collection_name} from {self.db_name} db ")

    def list_collection(self):
        """List the collections present in the vector store db

        Returns:
            List: List of collections in the vectore store
        """
        return self.chromaDB.list_collections()

    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection

        Args:
            collection_name (str): collection name

        Returns:
            Collection
        """

        return self.chromaDB.get_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def collection_exists(self, collection_name: str) -> bool:
        """Verify if a collection exists in vector store db

        Args:
            collection_name (str): Name of the collection to verify

        Returns:
            Bool
        """
        if collection_name is None:
            raise ValueError("Collection  is missing")

        existing_collections = [col.name for col in self.chromaDB.list_collections()]

        if collection_name in existing_collections:
            return True
        else:
            return False

    def add_doc_to_collection(
        self, docs: list[Document] | list[CoolChunk], collection: Collection
    ):
        """
        Add only the documents that are not present to a collection.

        Raises:
            ValueError: if no documents have been given
        """
        if docs is None:
            raise ValueError("No docs have been detected")

        # Verify if the user is using a docling processor or a classical one
        _using_docling = False
        if isinstance(docs[0], CoolChunk):
            _using_docling = True
            print("===== You are using Docling and some CoolChunks =====")

        if isinstance(docs[0], Document):
            print("===== You are using a langchain loader and chunker =====")

        # Unique ids
        ids = generate_ids(docs)

        # Actual ids present in the collection
        current_ids = set(collection.get(include=[])["ids"])

        current_ids = set(current_ids) if current_ids else set()

        # add the only doc that are not already in the collection
        doc_to_add = [doc for doc, id_ in zip(docs, ids) if id_ not in current_ids]
        ids_to_add = [id_ for id_ in ids if id_ not in current_ids]

        # Verify if there are documents to add
        if len(doc_to_add) > 0:
            try:
                if _using_docling:

                    # We need to serialize the metadas to string because chroma does not support lists
                    collection.add(
                        ids=ids_to_add,
                        documents=[doc.contextualized_chunk for doc in doc_to_add],
                        metadatas=[
                            {
                                "bboxes": json.dumps(d.bboxes),
                                "n_pages": json.dumps(d.n_pages),
                                "file_origin": d.origin,
                            }
                            for d in doc_to_add
                        ],
                    )
                else:
                    collection.add(
                        ids=ids_to_add,
                        documents=[doc.page_content for doc in doc_to_add],
                    )

                print("Documents successfully added to the collection")
            except Exception as e:
                print(" Error while adding documents to the collection : ", e)

        else:
            print("Nothing to add the collection already contains all the documents")

    def retrieve_docs(self, query: str, collection: Collection, top_k: int = 5) -> list:
        """Retrieves top_k documents from your collection that are in relation to the query

        Returns:
            list: list of list that contains relevant text
        """

        embedded_query = self.embedding_function.embed_query(query)
        try:
            docs = collection.query(query_embeddings=embedded_query, n_results=top_k)
            return docs["documents"]
        except Exception as e:
            print("Error while retrieving docs from the collection : ", e)

    def retrieve_docs_with_metadata(
        self, query: str, collection: Collection, top_k: int = 5
    ) -> dict:
        """Retrieves top_k documents from your collection that are in relation to the query
            with metadata
        Returns:
            list: list of list that contains relevant text
        """

        embedded_query = self.embedding_function.embed_query(query)
        try:
            docs = collection.query(query_embeddings=embedded_query, n_results=top_k)
            return {"texts": docs["documents"], "metadata": docs["metadatas"]}
        except Exception as e:
            print("Error while retrieving docs from the collection : ", e)
