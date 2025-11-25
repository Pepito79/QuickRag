from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from .VectorStore import VectorStore
from QuickRag.utils.api_key_verifier import verify_gemini_key
from langchain_classic.prompts import ChatPromptTemplate
import uuid
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 

class QuickRag:

    """
    Class to create a quick rag 
    """
    def __init__(self,
                 tokenizer:str = 'intfloat/multilingual-e5-small',
                 ):

        self.tokenizer_model = tokenizer
        
    def load_pdf_docs(self,
                      path: str) -> list[Document]:
        
        """Load pdf documents

        Raises:
            ValueError: if documents not loaded 
        Returns:
            list[Document]: chunks returned
        """
        
        loader = DirectoryLoader(
            path= path,
            glob="**/*.pdf",
            use_multithreading=True,
            recursive=True,
            show_progress=False
        )
        
        docs = loader.load()
        
        if not docs:
            raise ValueError("The loader failed to load your documents")
        
        else:
            print("Documents loaded successfully")
            return docs 
        

    def split_docs(self,
                   docs: list[Document],
                   chunk_size : int = 512,
                   ) -> list[Document]:
        
        """ Split the documents into chunks, with a recursivce splitter and  
        a tokenizer for counting the length .

        Args:
            docs (list[Document]): Docs to chunk
            chunk_size (int, optional): Size of the chunks. Defaults to 512.

        Returns:
            list[Document]: List of chunks where every chunk is a document
        """
        
        if docs is None:
            raise ValueError("No documents have been provided")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_model
        )
        
        # Use the tokenizer from hugging fae to count length 
        textsplitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            separators=["\n\n", "\n", " ", "","\n\n"],
            chunk_size = chunk_size,
            chunk_overlap = int(chunk_size/10),
            add_start_index = True,               
            strip_whitespace = True 
        )
        
        if textsplitter is None:
            raise ValueError("Failed to create the text splitter")
        
        #Chunks the document and times it 
        print("Chunking started ..... ")
        start = time.time()
        chunks = textsplitter.split_documents(docs)
        end = time.time()
        print(f"Chunking took  {end - start:.4f} seconds ")
        
        if chunks is None:
            raise ValueError('Documents have not been chunked ')
        
        return chunks
    
    
    def create_template(self,
                        prompt:str):
        
        return ChatPromptTemplate.from_template(prompt)
    
    
    def simple_rag_gemini(self,
                   path_documents:str,
                   query:str,
                   gemini_model:str,
                   collection_name: str | None,
                   path_db:str  = "./chromadb",):
        
        
        #Verify if the gemini api key is in the .env
        verify_gemini_key()
        #If no collection name we create an unique one
        if collection_name is None:
            col_name = str(uuid.uuid4())
        else:
            col_name = collection_name
        
        db_name  = str(uuid.uuid4())
        
        # Create the vector store database
        try:    
            db = VectorStore(db_name,path_db)
            print("Vector database succesfully created ...\n ")
        except Exception as e:
            print("Exception raised while trying to create the vector database : ",e)
            
        
        #Load the documents
        try:
            docs = self.load_pdf_docs(path_documents)
            print("Documents succesfully loaded ...\n ")
        except Exception as e:
            print("Exception raised while trying to load the documents : ",e)
             
        
        #Chunk the documents
        try:
            chunks = self.split_docs(docs,512)
            print("Documents succesfully chunked ...\n ")
            
        except Exception as e:
            print("Exception raised while trying to chunks the documents : ",e)
            

        #Create or get the collection if it already exists
        emb_model = "intfloat/multilingual-e5-small"
        try:
            collection = db.create_collection(col_name,emb_model)
        except Exception as e:
            print("Exception raised while trying to create/get the collection : ",e)
        
        #Add the documents to the collection
        db.add_doc_to_collection(chunks,collection)

        # Retrieves the top 5 document (default value)
        retrieved_docs = db.retrieve_docs(query,collection,top_k=5)        
        print("Doc retrieved successfully\n")
        
        
        template = """Use the following context to answer the question at the end. 
           You must be respectful and helpful, and answer in the language of the question.
           If you don't know the answer, say that you don't know.

           Context: {context}

           Question: {question}
           """
        
        # Create the runnables to make a chain 
        prompt = self.create_template(prompt=template)  
        prompt_runnable = RunnableLambda(lambda args : prompt.format_messages(context = args['context'] , question = args['question']))
        context_runnable = RunnableLambda(lambda _ :"\n\n".join(doc[0] for doc in retrieved_docs))
        query_runnable= RunnablePassthrough()
        llm = ChatGoogleGenerativeAI(
            model= gemini_model,
            temperature=0.6,
        )
        
        pipeline = (
            {
                "context":context_runnable,
                "question": query_runnable
            }
            | prompt_runnable
            | llm
            | StrOutputParser()
        )
        
        answer = pipeline.invoke(query)
        return answer
        
        

            
        
        
        