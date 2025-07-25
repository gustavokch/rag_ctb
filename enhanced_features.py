import os
import dotenv
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.client import configure
from typing import List
from rag_pipeline import SearchResult # Assuming rag_pipeline.py is in the same directory

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure GEMINI_API_KEY is set from environment variables

class EnhancedFeatures:
    def __init__(self):
        configure(api_key=GEMINI_API_KEY)

    def expand_query(self, original_query: str) -> List[str]:
        """Generate related queries for better retrieval"""
        expansion_prompt = f"""
        Generate 3 alternative phrasings of this legal query that might help find relevant information:
        Original: {original_query}
        
        Alternatives:
        """
        
        model = GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(expansion_prompt)
        
        # Parse alternatives and combine results
        return [original_query] + response.text.strip().split('\n')

    def verify_citations(self, answer: str, sources: List[SearchResult]) -> dict:
        """Verify that citations in answer match sources"""
        verification_prompt = f"""
        Verify that the following answer is supported by the provided sources.
        Rate support level from 0-1.
        
        Answer: {answer}
        
        Sources: {[s.content[:200] + "..." for s in sources]}
        
        Verification score (0-1):
        """
        
        model = GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(verification_prompt)
        
        try:
            score = float(response.text.strip())
            return {'verification_score': min(max(score, 0), 1)}
        except:
            return {'verification_score': 0.5}