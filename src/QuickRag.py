from langchain_core.documents import Document
from utils.spliter import split_docs
from langchain_google_genai import ChatGoogleGenerativeAI
from VectorStore import VectorStore
from utils.api_key_verifier import verify_gemini_key
from langchain_classic.prompts import ChatPromptTemplate
import uuid
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from Reranker import Reranker
from utils.loader import load_pdf_docs

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
        reranker: bool,
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

        # Verify if we using a reranker
        if reranker:
            context_runnable = RunnableLambda(
                lambda _: "\n\n".join(doc for doc in retrieved_docs)
            )
        else:
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
        reranker: Reranker | None = None,
        collection_name: str | None = None,
        path_db: str = "./chromadb",
    ):
        """Create a the easiest rag that use gemini and where you can use a reranker

        Args:
            path_documents (str): the documents path
            query (str): user's query
            gemini_model (str): the gemini model
            reranker (Reranker | None, optional): An optionnal reranker to have better result. Defaults to None.
            collection_name (str | None, optional): name of the collection to create. Defaults to None.
            path_db (str, optional): _description_. path where to store the db to "./chromadb".

        Returns:
            _type_: _description_
        """

        # Verify if the gemini api key is in the .env
        verify_gemini_key()

        # If no collection name we create an unique one
        if collection_name is None:
            col_name = str(uuid.uuid4())
        else:
            col_name = collection_name

        db_name = str(uuid.uuid4())

        # Create the vector store database
        try:
            db = VectorStore(db_name, path_db)
            if os.path.exists(path_db):
                print("Vector database already exists ...\n ")
            else:
                print("Vector database successfully created ...\n")
        except Exception as e:
            print("Exception raised while trying to create the vector database : ", e)

        # Load the documents
        try:
            docs = load_pdf_docs(path_documents)
        except Exception as e:
            print("Exception raised while trying to load the documents : ", e)

        # Chunk the documents
        try:
            chunks = split_docs(self.tokenizer_model, docs, 512)
            print("Documents succesfully chunked ...\n ")

        except Exception as e:
            print("Exception raised while trying to chunks the documents : ", e)

        # Create or get the collection if it already exists
        emb_model = "intfloat/multilingual-e5-small"
        try:
            if db.collection_exists(col_name):
                print(
                    f"The collection {col_name} already exists nothing will be created ....\n"
                )
            collection = db.create_collection(col_name, emb_model)
            print(f"Collection {col_name} successfully created ....\n")

        except Exception as e:
            print("Exception raised while trying to create/get the collection : ", e)

        # Add the documents to the collection
        db.add_doc_to_collection(chunks, collection)

        # Retrieves the top 5 document (default value)
        retrieved_docs = db.retrieve_docs(query, collection, top_k=5)
        print("Doc retrieved successfully\n")

        # Use the reranker if it passed as a parameter
        if reranker is not None:

            # Put all the text in one list to give it to the reranker
            documents_in_retrieves_docs = [
                doc_text for doc in retrieved_docs for doc_text in doc
            ]

            # Ranks the texts
            reranked_doc = reranker.rank_docs(
                query=query, docs=documents_in_retrieves_docs
            )

            context = reranked_doc

        else:
            context = retrieved_docs

        answer = self.get_answer_gemini(
            query=query,
            gemini_model=gemini_model,
            retrieved_docs=context,
            reranker=True,
        )

        return answer
