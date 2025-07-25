import os
import dotenv
import pymupdf as fitz
import chromadb
import numpy as np
from chromadb.config import Settings
from typing import List, Dict
from google.generativeai.client import configure
from google.generativeai.embedding import embed_content


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
        configure(api_key=GEMINI_API_KEY)
        
    def embed_text(self, text: str) -> np.ndarray: # Explicitly type return as np.ndarray
        """Generate embeddings using Gemini"""
        response = embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response['embedding'])
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to vector database"""
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        embeddings: List[np.ndarray] = [np.array(self.embed_text(text)) for text in texts] # Explicitly type for Pylance
        
        self.collection.add(
            embeddings=embeddings,
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
