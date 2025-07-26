import os
import dotenv
import pymupdf as fitz
import chromadb
import numpy as np
from chromadb.config import Settings
from typing import List, Dict
from google import genai
from google.genai import types
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegislationVectorDB:
    def __init__(self, db_path: str = "./legislation_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="legislation_chunks",
            metadata={"description": "Legislation document chunks"}
        )
        dotenv.load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # FIX: Corrected environment variable access
        # Configure Gemini for embeddings
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        
    def embed_text(self, text: str) -> np.ndarray: # Explicitly type return as np.ndarray
        """Generate embeddings using Gemini"""
        client = self.genai_client
        try:
            embed_results = client.models.embed_content(
                    model="text-embedding-004",
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            
            if not embed_results or not embed_results.embeddings:
                logger.error(f"Embeddings not found in embed_results for text: {text[:50]}...")
                raise ValueError("Failed to generate embeddings: No embeddings returned.")
            
            # Ensure embeddings is a list of numerical values
            # The error indicates that e.values might already be a list of numbers,
            # and np.array(e.values) is creating a 1D array.
            # The issue is likely that embed_results.embeddings is a list of these 1D arrays,
            # and then np.array(response) is making it a 2D array of arrays.
            
            # We expect a list of 1D arrays (or lists of floats) for ChromaDB.
            # If embed_results.embeddings contains multiple embedding objects,
            # and each e.values is a list of floats, then `response` will be a list of 1D numpy arrays.
            # This is the format ChromaDB expects for the `embeddings` parameter when adding multiple embeddings.
            
            # If we are embedding a single text, embed_results.embeddings should contain one embedding object.
            # In that case, we want to return a single 1D numpy array.
            
            # Let's assume embed_results.embeddings is a list of embedding objects,
            # and each e.values is a list of floats.
            
            # For a single text input, embed_results.embeddings will be a list containing one embedding object.
            # We want to extract the values from this single embedding object as a 1D numpy array.
            result = np.array([e.values for e in embed_results.embeddings], dtype=np.float32)
            if result.ndim == 2:
                # If we have a 2D array (multiple embeddings), we need to flatten it.
                result = result.flatten()
            logger.info(f"DEBUG: Embedding generated with shape {result.shape}")
            if result.ndim == 1:
                logger.info(f"DEBUG: Successfully generated 1D embedding.")
            else:
                logger.error(f"Unexpected shape for embedding: {result.shape}. Expected 1D.")
                raise ValueError("Unexpected embedding shape from embed_text")

            logger.info(f"DEBUG: Type of result before returning from embed_text: {type(result)}")
            if isinstance(result, np.ndarray):
                logger.info(f"DEBUG: Shape of result (np.ndarray): {result.shape}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating embeddings for text: {text[:50]}... Error: {e}")
            raise

    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to vector database"""
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings. embed_text now consistently returns a 1D np.ndarray.
        # Pass the list of 1D np.ndarray objects directly to ChromaDB.
        embeddings_for_chroma: List[np.ndarray] = [self.embed_text(text) for text in texts]
        
        # Debugging: Log the type and shape of the embeddings list before adding to ChromaDB
        logger.info(f"DEBUG: Type of embeddings list before add: {type(embeddings_for_chroma)}")
        if len(embeddings_for_chroma) > 0:
            logger.info(f"DEBUG: Type of first embedding in list: {type(embeddings_for_chroma[0])}")
            if isinstance(embeddings_for_chroma[0], np.ndarray):
                logger.info(f"DEBUG: Shape of first embedding in list: {embeddings_for_chroma[0].shape}")
            
        self.collection.add(
            embeddings=embeddings_for_chroma,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, n_results: int = 5):
        """Search for relevant chunks"""
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
