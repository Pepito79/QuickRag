from dotenv import load_dotenv
import os


def verify_gemini_key():
    
    load_dotenv()
    GOOGLE_KEY = "GOOGLE_API_KEY"
    
    api_key_value = os.environ.get(GOOGLE_KEY)
    if not api_key_value:
        raise ValueError('There is no gemini key in your env !')
        exit(1)
    else:
        print('Gemini key loaded ')
    
    