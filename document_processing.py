import pymupdf as fitz  # PyMuPDF - better for complex PDFs
import re
from typing import List, Dict

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure GEMINI_API_KEY is set from environment variables


def extract_text_from_pdf(pdf_path):
    """Extract text while preserving structure"""
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_textpage().extractText()        
        # Preserve page numbers for citations
        full_text.append({
            'page': page_num + 1,
            'content': text,
            'metadata': {
                'source': pdf_path,
                'page_number': page_num + 1
            }
        })
    
    return full_text

def chunk_legislation_text(pages: List[Dict], chunk_size: int = 1000, overlap: int = 200):
    """Smart chunking for legal documents"""
    chunks = []
    
    for page_data in pages:
        text = page_data['content']
        page_num = page_data['page']

        # Split by legal sections (Artigo, Secção, Parágrafo)
        section_pattern = r'(CAPÍTULO|SEÇÃO|Seção|ARTIGO|Art\.|Artigo\s+\d+|Seção\s+\d+|§\s*\d+|\d+\.\s|\(\d+\))'
        sections = re.split(section_pattern, text)
        
        current_chunk = ""
        current_metadata = {
            'source_page': page_num,
            'chunk_type': 'content'
        }
        
        for i, section in enumerate(sections):
            if len(current_chunk + section) > chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': current_metadata.copy()
                    })
                current_chunk = section
            else:
                current_chunk += section
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': current_metadata
            })
    print(len(chunks), "chunks created from legislation text.")
    return chunks