from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class Reranker:
    
    
    def __init__(self,
                 model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 ):
        
        self.cross_encoder = CrossEncoder(model)
        
        
    def rank_docs(self,
                  query: str,
                  docs: list[str],
                  top_k: int = 5):
        
        
        # Examine all the possible issues
        if query is None:
            raise ValueError('No query has been given')
        if docs is None:
            raise ValueError('No documents have been given')
        if len(docs) == 0 :
            raise ValueError('The list of documents is empty ')
        
        try:
            ranked_docs = self.cross_encoder.rank(query,docs,top_k)
        except Exception as e:
            print('Exception raised while trying to rank the documents :\n',e)
        return ranked_docs
        
        