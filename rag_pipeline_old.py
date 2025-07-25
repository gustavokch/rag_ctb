import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, cast # Added Dict, Any, cast
import numpy as np
import google.generativeai as genai
import google.generativeai.client as genai_client
import google.generativeai.generative_models as genai_models

configure = genai_client.configure
from vector_database import LegislationVectorDB # Assuming vector_database.py is in the same directory

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure GEMINI_API_KEY is set from environment variables


@dataclass
class SearchResult:
    content: str
    page: int
    confidence: float
    metadata: Dict[str, Any] # Changed to Dict[str, Any]

class LegislationRAG:
    def __init__(self, vector_db: LegislationVectorDB):
        self.vector_db = vector_db
        configure(api_key=GEMINI_API_KEY)
        
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve most relevant chunks"""
        results = self.vector_db.search(query, n_results=top_k)
        logging.info(f"Raw search results: {results}")
        logging.info(f"Type of results: {type(results)}")
        if isinstance(results, dict):
            logging.info(f"Keys in results: {results.keys()}")
            docs = results.get('documents')
            metas = results.get('metadatas')
            dists = results.get('distances')
            logging.info(f"results['documents']: {docs}, Type: {type(docs)}")
            logging.info(f"results['metadatas']: {metas}, Type: {type(metas)}")
            logging.info(f"results['distances']: {dists}, Type: {type(dists)}")

            if docs is not None and len(docs) > 0: # Check for None before len
                logging.info(f"results['documents'][0]: {docs[0]}, Type: {type(docs[0])}")
            if metas is not None and len(metas) > 0: # Check for None before len
                logging.info(f"results['metadatas'][0]: {metas[0]}, Type: {type(metas[0])}")
            if dists is not None and len(dists) > 0: # Check for None before len
                logging.info(f"results['distances'][0]: {dists[0]}, Type: {type(dists[0])}")
        else:
            logging.warning("Search results are not a dictionary.")
        
        search_results = []
        
        # Check if documents, metadatas, and distances are not None and have at least one element
        documents = results.get('documents')
        metadatas = results.get('metadatas')
        distances = results.get('distances')

        if documents and len(documents) > 0 and \
           metadatas and len(metadatas) > 0 and \
           distances and len(distances) > 0:
            
            # Ensure the first elements exist before zipping
            if documents[0] is not None and metadatas[0] is not None and distances[0] is not None:
                for i, (doc, metadata, distance) in enumerate(zip(
                    documents[0],
                    metadatas[0],
                    distances[0]
                )):
                    confidence = 1 - distance  # Convert distance to confidence
                    
                    # Log the type of metadata and source_page
                    logging.info(f"Type of metadata: {type(metadata)}")
                    source_page_value = metadata.get('source_page', 0)
                    logging.info(f"Value of source_page: {source_page_value}, Type: {type(source_page_value)}")

                    # Ensure source_page_value is convertible to int
                    page_int = 0
                    if isinstance(source_page_value, (int, float)):
                        page_int = int(source_page_value)
                    elif isinstance(source_page_value, str) and source_page_value.isdigit():
                        page_int = int(source_page_value)
                    
                    search_results.append(SearchResult(
                        content=doc,
                        page=page_int, # Explicitly cast to int and handle non-int types
                        confidence=confidence,
                        metadata=cast(Dict[str, Any], metadata) # Explicitly cast Metadata to Dict[str, Any]
                    ))
            else:
                logging.warning("First elements of 'documents', 'metadatas', or 'distances' are empty or None.")
        else:
            logging.warning("One or more of 'documents', 'metadatas', or 'distances' are missing or empty in search results.")
            
        return search_results
    
    def generate_answer(self, query: str, context_chunks: List[SearchResult]) -> dict:
        """Generate answer using Gemini with retrieved context"""
        
        # Prepare context
        context = "\n\n".join([
            f"[Page {chunk.page}] {chunk.content}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        prompt = f"""
You are a legal expert analyzing legislation. Based on the provided context from the legislation document, answer the question accurately and precisely.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide a precise, accurate answer based ONLY on the provided context, in Brazilian Portuguese.
2. Include specific citations with page numbers
3. If the answer cannot be fully determined from the context, state this clearly
4. Maintain legal terminology accuracy
5. Structure your response clearly

ANSWER:
"""

        try:
            model = genai_models.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            
            return {
                'answer': response.text,
                'sources': [{'page': chunk.page, 'confidence': chunk.confidence}
                           for chunk in context_chunks],
                'status': 'success'
            }
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'status': 'error'
            }
    
    def calculate_confidence_score(self, query: str, answer: str, sources: List[dict]) -> float:
        """Calculate overall confidence score"""
        # Weighted average of source confidences
        if not sources:
            return 0.0
        
        source_confidence = np.mean([s['confidence'] for s in sources])
        logging.info(f"Type of source_confidence: {type(source_confidence)}")
        logging.info(f"Value of source_confidence: {source_confidence}")
        
        # Additional factors could include:
        # - Answer length appropriateness
        # - Keyword overlap between query and sources
        # - Consistency across multiple sources
        
        return float(min(float(source_confidence), 1.0)) # Explicitly cast source_confidence to float
    
    def ask_question(self, question: str) -> dict:
        """Main interface for asking questions"""
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=5)
        
        # Generate answer
        result = self.generate_answer(question, relevant_chunks)
        
        # Calculate confidence
        confidence = self.calculate_confidence_score(
            question,
            result['answer'],
            result['sources']
        )
        
        return {
            'question': question,
            'answer': result['answer'],
            'citations': result['sources'],
            'confidence_score': confidence,
            'retrieved_chunks': len(relevant_chunks)
        }