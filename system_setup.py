from document_processing import extract_text_from_pdf, chunk_legislation_text
from vector_database import LegislationVectorDB
from rag_pipeline import LegislationRAG
import subprocess
import sys
import time

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure GEMINI_API_KEY is set from environment variables

def setup_legislation_rag_system(pdf_path: str):
    """Complete setup pipeline"""
    
    print("1. Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    
    print("2. Chunking document...")
    chunks = chunk_legislation_text(pages)
    print(f"Created {len(chunks)} chunks")
    
    print("3. Setting up vector database...")
    vector_db = LegislationVectorDB()
    
    print("4. Generating embeddings and storing...")
    if os.path.exists(".is_db_initialized"):
        print("Vector database already initialized. Skipping embedding generation.")
    else:
        print("Adding chunks to vector database...")
        vector_db.add_chunks(chunks)
    with open(".is_db_initialized", "w") as f:
        f.write("Vector database initialized.")
    
    print("5. Initializing RAG system...")
    rag_system = LegislationRAG(vector_db)
    
    print("✅ System ready!")
    return rag_system

# Usage example (commented out for now, as it requires a PDF file)
if __name__ == "__main__":
    rag_system = setup_legislation_rag_system("ctb.pdf")
    result = rag_system.ask_question("What are the penalties for violation of Article 15?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Citations: {result['citations']}")

#Example usage for deployment (uncomment to use):
# if __name__ == "__main__":
#     deploy_streamlit_with_cloudflared("web_interface.py", port=35333)