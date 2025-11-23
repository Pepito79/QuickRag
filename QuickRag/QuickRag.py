from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from .VectorStore import VectorStore

class QuickRag:

    """
    Class to create a quick rag 
    """
    def __init__(self,
                 tokenizer:str = 'intfloat/multilingual-e5-small',
                 ):
        self.tokenizer_model = tokenizer
    
    def load_pdf_docs(self,
                      path: str):
        
        """Load pdf documents

        Raises:
            ValueError: if documents not loaded 
        Returns:
            type_: List[Documents]
        """
        
        loader = DirectoryLoader(
            path= path,
            glob="**/*.pdf",
            use_multithreading=True,
            recursive=True,
            show_progress=True
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
        
