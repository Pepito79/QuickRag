from langchain_core.documents import Document
from utils.spliter import split_docs
from langchain_google_genai import ChatGoogleGenerativeAI
from VectorStore import VectorStore
from utils.api_key_verifier import verify_gemini_key
from langchain_classic.prompts import ChatPromptTemplate
import uuid
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from Reranker import Rerankers
from utils.loader import load_pdf_docs
from Docling.DoclingProcessor import DoclingProcessor
from utils.box_drawer import draw

import os


class QuickRag:
    """
    Class to create a quick rag
    """

    def __init__(
        self,
        tokenizer: str = "intfloat/multilingual-e5-small",
    ):

        self.tokenizer_model = tokenizer

    def get_answer_gemini(
        self,
        query: str,
        gemini_model: str,
        retrieved_docs: list[list],
    ):

        # Verify if the gemini api key is in the .env
        verify_gemini_key()

        # Prompt template to use
        template = """Use the following context to answer the question at the end. 
           You must be respectful and helpful, and answer in the language of the question.
           If you don't know the answer, say that you don't know.

           Context: {context}

           Question: {question}
           """

        prompt = ChatPromptTemplate.from_template(template)
        prompt_runnable = RunnableLambda(
            lambda args: prompt.format_messages(
                context=args["context"], question=args["question"]
            )
        )

        context_runnable = RunnableLambda(
            lambda _: "\n\n".join(doc for doc in retrieved_docs)
        )

        query_runnable = RunnablePassthrough()
        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=0.6,
        )

        pipeline = (
            {"context": context_runnable, "question": query_runnable}
            | prompt_runnable
            | llm
            | StrOutputParser()
        )

        answer = pipeline.invoke(query)
        return answer

    def create_naive_gemini(
        self,
        path_documents: str,
        query: str,
        gemini_model: str,
        useDocling: bool = True,
        useVLM: bool = False,
        useOCR: bool = False,
        highlight: bool = False,
        reranker: Rerankers | None = None,
        collection_name: str | None = None,
        path_db: str = "./chromadb",
    ):
        """Create a the easiest rag that use gemini and where you can use a reranker

        Args:
            path_documents (str): the documents path
            query (str): user's query
            gemini_model (str): the gemini model
            reranker (Rerankers | None, optional): An optionnal reranker to have better result. Defaults to None.
            collection_name (str | None, optional): name of the collection to create. Defaults to None.
            path_db (str, optional): _description_. path where to store the db to "./chromadb".

        Returns:
            _type_: _description_
        """

        # Verify if the gemini api key is in the .env
        verify_gemini_key()

        if highlight and not useDocling:
            raise Exception(
                " Impossible de highlight without using docling , feature coming soon ......"
            )

        # If no collection name we create an unique one
        if collection_name is None:
            col_name = str(uuid.uuid4())
        else:
            col_name = collection_name

        db_name = str(uuid.uuid4())

        # Create the vector store database
        db = VectorStore(db_name, path_db)
        print("Database successfully initialized")
        if os.path.exists(path_db):
            print("Vector database already exists ...\n ")
        else:
            print("Vector database successfully created ...\n")

        # Case without docling
        if not useDocling:
            try:
                docs = load_pdf_docs(path_documents)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load documents from {path_documents}"
                ) from e

            if not docs:
                raise ValueError("No documents were loaded")

            try:
                chunks = split_docs(self.tokenizer_model, docs, 512)
                print("Documents successfully chunked\n")
            except Exception as e:
                raise RuntimeError("Failed to split documents into chunks") from e

        else:
            try:
                docling_processor = DoclingProcessor(
                    paths=path_documents,
                    useOCR=useOCR,
                    useSmolVLM=useVLM,
                )
                chunks = docling_processor.process()
            except Exception as e:
                raise RuntimeError("Docling processing failed\n", e) from e

            if not chunks:
                raise ValueError("Docling returned no chunks")

        if db.collection_exists(col_name):
            print(
                f"The collection {col_name} already exists nothing will be created ....\n"
            )
            collection = db.get_collection(col_name)
        else:
            collection = db.create_collection(collection_name=col_name)
            print(f"Collection {col_name} successfully created ....\n")

        # Add the documents to the collection
        db.add_doc_to_collection(chunks, collection)
        print("Chunks added to the collection")
        # Retrieves the top 5 document (default value)

        if useDocling:
            retrieved_docs_with_meta = db.retrieve_docs_with_metadata(
                query, collection, top_k=5
            )

            retrieved_text = retrieved_docs_with_meta["texts"]
            print("Doc retrieved successfully with their metadata \n")

        else:
            retrieved_text = db.retrieve_docs(query, collection, top_k=5)
            print("Doc retrieved successfully  \n")

        # Use the reranker if it passed as a parameter
        if reranker is not None:

            # Put all the text in one list to give it to the reranker
            documents_in_retrieves_docs = [
                doc_text for doc in retrieved_text for doc_text in doc
            ]

            # Ranks the texts
            reranked_doc = reranker.rank_docs(
                query=query, docs=documents_in_retrieves_docs
            )
            print("Documents ranked")

            context = reranked_doc

        else:
            context = retrieved_text

        if isinstance(context[0], list):
            context = [d for sub in context for d in sub]

        answer = self.get_answer_gemini(
            query=query, gemini_model=gemini_model, retrieved_docs=context
        )

        if highlight:
            draw(retrieved_docs_with_meta)

        return answer
