from langchain_chroma import Chroma
from chromadb import PersistentClient
from chromadb.types import Collection
from langchain_core.documents import Document
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from QuickRag.utils.hash_functions import generate_ids
import torch
from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import create_langchain_embedding

class  VectorStore:
    """
    VectorStore class that stocks collections and allows to get access / modify / delete
    and add knowledge to it.
    """
    def __init__(self,
                 db_name: str,
                 persistent_dir: str = "./chromDB",
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
        
        
        
    def create_collection(self,
                       collection_name:str,
                       embedding_model_name: Optional[str] = None) -> Collection:
        
        """
        If the collection is not in the db it creates one with an embedding model coming from langchain 
        and if it already exists it returns it.
        
        Returns:
            Collection
        """
        
        #If the user does not give any embedding model we use the default one of chroma
        if embedding_model_name is None:
            self.embedding_model = None
            
        #Pull the model from HuggingFace
        else:
            
            # Find the available device
            if torch.cuda.is_available():
                args= {"device":"cuda"}
            else:
                args = {"device":"cpu"}
            self.embedding_model =  create_langchain_embedding(HuggingFaceEmbeddings(model_name=embedding_model_name))
        

        # Create the collection
        try:
            collection = self.chromaDB.get_or_create_collection(
                name= collection_name,
                embedding_function=self.embedding_model
            )

        except Exception as e:
            print("Exception raised\n",e)
            return
        
        print("Collection created successfully !")
        return  collection
    
    def delete_collection(self,
                          collection_name:str):
        """
        Delete an existing collection from the vector store db

        Args:
            collection_name (str): name of the collection to delete
        """
        self.chromaDB.delete_collection(
            collection_name
        )
        
        print(f"Deleted {collection_name} from {self.db_name} db ")

     
    def list_collection(self):
        """List the collections present in the vector store db

        Returns:
            List: List of collections in the vectore store
        """
        return self.chromaDB.list_collections()
    
    def get_collection(self,
                       collection_name:str) -> Collection:
        """Get a collection

        Args:
            collection_name (str): collection name

        Returns:
            Collection
        """
        return self.chromaDB.get_collection(
            name=collection_name
        )
        
    def collection_exists(self, collection: Collection) -> bool:
        
        """ Verify if a collection exists in vector store db

        Args:
            collection_name (str): Name of the collection to verify

        Returns:
            Bool
        """
        if collection is None:
            raise ValueError("Collection  is missing")
        
        existing_collections = [col for col in self.chromaDB.list_collections()]

        if collection in existing_collections:
            return True
        else:
            return False
        
        
    def add_doc_to_collection(self,
                              docs: list[Document],
                              collection: Collection 
                              ):
        
        """
        Add only the documents that are not present to a collection. 

        Raises:
            ValueError: if no documents have been given
        """
        if docs is None:
            raise ValueError('No docs have been detected')
        
        
        #Unique ids 
        ids = generate_ids(docs)
        
        #Actual ids present in the collection
        current_ids = set(
            collection.get(include=[])["ids"]
        )

        current_ids = set(current_ids) if current_ids else set()
       
        #add the only doc that are not already in the collection
        doc_to_add = [doc for doc, id_ in zip(docs, ids) if id_ not in current_ids]
        ids_to_add = [id_ for id_ in ids if id_ not in current_ids]
        
        #Verify if there are documents to add
        if len(doc_to_add)>0:
            try:
                collection.add(
                    ids = ids_to_add,
                    documents = [doc.page_content for doc in doc_to_add]
                )
                return collection
            except Exception as e:
                print(" Error while adding documents to the collection : ",e)
        
        else:
            print("Nothing to add the collection already contains all the documents")
        
        
    