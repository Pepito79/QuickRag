from langchain_google_genai import ChatGoogleGenerativeAI



class Gemini:
    
    def __init__(self,
                 model_name: str):
        self.model = ChatGoogleGenerativeAI(
            model = model_name,
            temperature = 0.6,
            max_retries=  3
        )
    
    def answer(self,query:str):
        
        return self.model.invoke(query)