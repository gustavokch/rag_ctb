import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, cast
import numpy as np
import google.genai as genai
from google.genai import types
import google.genai.client as genai_client
import google.genai.models as genai_models

from vector_database import LegislationVectorDB

import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@dataclass
class SearchResult:
    content: str
    page: int
    confidence: float
    metadata: Dict[str, Any] # Changed to Dict[str, Any]

class LegislationRAG:
    def __init__(self, vector_db: LegislationVectorDB):
        self.vector_db = vector_db
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.logger = logging.getLogger(__name__)
        self.logger.level = logging.ERROR

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve most relevant chunks"""
        results = self.vector_db.search(query, n_results=top_k)
        if isinstance(results, dict):
            self.logger.info(f"Keys in results: {results.keys()}")
            docs = results.get('documents')
            metas = results.get('metadatas')
            dists = results.get('distances')
            self.logger.info(f"results['documents']: {docs}, Type: {type(docs)}")
            self.logger.info(f"results['metadatas']: {metas}, Type: {type(metas)}")
            self.logger.info(f"results['distances']: {dists}, Type: {type(dists)}")

            if docs is not None and len(docs) > 0: # Check for None before len
                self.logger.info(f"results['documents'][0]: {docs[0]}, Type: {type(docs[0])}")
            if metas is not None and len(metas) > 0: # Check for None before len
                self.logger.info(f"results['metadatas'][0]: {metas[0]}, Type: {type(metas[0])}")
            if dists is not None and len(dists) > 0: # Check for None before len
                self.logger.info(f"results['distances'][0]: {dists[0]}, Type: {type(dists[0])}")
        else:
            self.logger.warning("Search results are not a dictionary.")
        
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
                    self.logger.info(f"Type of metadata: {type(metadata)}")
                    source_page_value = metadata.get('source_page', 0)
                    self.logger.info(f"Value of source_page: {source_page_value}, Type: {type(source_page_value)}")

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
                self.logger.warning("First elements of 'documents', 'metadatas', or 'distances' are empty or None.")
        else:
            self.logger.warning("One or more of 'documents', 'metadatas', or 'distances' are missing or empty in search results.")
            
        return search_results
    
    def generate_answer(self, query: str, context_chunks: List[SearchResult], max_tokens: Optional[int]) -> dict:
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
1. Provide a precise, accurate answer based on the provided context, in Brazilian Portuguese.
2. If the question includes multiple choice options, select the most appropriate one based on the context and start your response with "A resposta correta é:"
3. Include specific citations with page numbers
4. If the answer cannot be fully determined from the context, state this clearly
5. Maintain legal terminology accuracy
6. Structure your response clearly

ANSWER:
"""

        try:
            model = 'gemini-2.5-flash'
            response = self.genai_client.models.generate_content(model=model, contents=prompt, config= types.GenerateContentConfig(temperature=0.1, max_output_tokens=4096))

            # self.logger.info(f"generate_answer: Creating sources from context_chunks. Example chunk.page type: {type(context_chunks[0].page) if context_chunks else 'N/A'}, chunk.confidence type: {type(context_chunks[0].confidence) if context_chunks else 'N/A'}")
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
    
    def generate_grounded_answer(self, model: str, query: str, context_chunks: Optional[List[SearchResult]], max_tokens: Optional[int]) -> types.GenerateContentResponse:
        """Generate a grounded answer using Google Search as a tool."""
        
        # Prepare context
        if context_chunks is not None:
            context = "CONTEXT:\n\n\n".join([
                f"[Page {chunk.page}] {chunk.content}"
                for chunk in context_chunks
            ])
        if context_chunks is None:
            context = ""
        

        # Create prompt
        prompt = f"""
        You are a legal expert analyzing legislation. Based on the provided context from the legislation document, answer the question accurately and precisely.

        {context}

        QUESTION: {query}

        INSTRUCTIONS:
        1. Provide a precise, accurate answer based on the web search, in Brazilian Portuguese.
        2. If the question includes multiple choice options, select the most appropriate one based on the context and ALWAYS start your response with "A resposta correta é:"
        3. Include specific citations
        4. If the answer cannot be fully determined from the context, state this clearly
        5. Maintain legal terminology accuracy
        6. Structure your response clearly

        ANSWER:
        """
        try:
            # Define the grounding tool
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
        
            # Configure generation settings
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens if max_tokens is not None else 4096,
                tools=[grounding_tool]
            )

            # Make the request
            response = self.genai_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response
        except Exception as e:
            self.logger.error(f"Error generating grounded response: {e}")
            # Return a mock object that mimics the expected structure for error cases
            class MockGenerateContentResponse:
                def __init__(self, text_content: str):
                    self.text = text_content
                    self.candidates = []
                    self.grounding_metadata = None # Add grounding_metadata attribute to prevent AttributeError

            return cast(types.GenerateContentResponse, MockGenerateContentResponse(f"Error generating grounded response: {str(e)}"))

    def calculate_confidence_score(self, query: str, answer: str, sources: List[dict]) -> float:
        """Calculate overall confidence score"""
        # Weighted average of source confidences
        if not sources:
            return 0.0
        
        source_confidence = np.mean([s['confidence'] for s in sources])
        self.logger.info(f"Type of source_confidence: {type(source_confidence)}")
        self.logger.info(f"Value of source_confidence: {source_confidence}")
        
        # Additional factors could include:
        # - Answer length appropriateness
        # - Keyword overlap between query and sources
        # - Consistency across multiple sources
        
        return float(min(float(source_confidence), 1.0)) # Explicitly cast source_confidence to float

    def expand_query(self, original_query: str) -> List[str]:
        """Generate related queries for better retrieval"""
        expansion_prompt = f"""
        Generate 3 alternative phrasings of this legal query that might help find relevant information:
        Original: {original_query}
        
        Alternatives:
        """
        
        model = 'gemini-2.5-flash'
        response = self.genai_client.models.generate_content(model=model, contents=expansion_prompt, config= types.GenerateContentConfig(temperature=0.1, max_output_tokens=4096))
       
        # Parse alternatives and combine results
        return [original_query] + (response.text.strip() if response.text is not None else "").split('\n')

    def verify_citations(self, answer: str, sources: List[SearchResult]) -> dict:
        """Verify that citations in answer match sources"""
        verification_prompt = f"""
        Verify that the following answer is supported by the provided sources.
        Rate support level from 0-1.
        
        Answer: {answer}
        
        Sources: {[s.content[:200] + "..." for s in sources]}
        
        Verification score (0-1):
        """
        
        model = 'gemini-2.5-flash'
        response = self.genai_client.models.generate_content(model=model, contents=verification_prompt, config= types.GenerateContentConfig(temperature=0.1, max_output_tokens=4096))

        try:
            score = float(response.text.strip() if response.text is not None else "0.5")
            return {'verification_score': min(max(score, 0), 1)}
        except:
            return {'verification_score': 0.5}

    def add_citations(self, response: types.GenerateContentResponse) -> Tuple[str, List[str]]:
        text = response.text if response.text is not None else ""
        citation_links = []

        # Remove previous logging
        # self.logger.info(f"add_citations: response.candidates: {response.candidates}")
        if not response.candidates:
            self.logger.warning("add_citations: response.candidates is empty or None.")
            return text, citation_links

        candidate = response.candidates[0]
        # self.logger.info(f"add_citations: candidate: {candidate}")

        if not hasattr(candidate, 'grounding_metadata') or candidate.grounding_metadata is None:
            self.logger.warning("add_citations: candidate.grounding_metadata is missing or None.")
            return text, citation_links

        grounding_metadata = candidate.grounding_metadata
        # self.logger.info(f"add_citations: grounding_metadata: {grounding_metadata}")

        supports = grounding_metadata.grounding_supports
        # self.logger.info(f"add_citations: supports: {supports}")
        chunks = grounding_metadata.grounding_chunks
        # self.logger.info(f"add_citations: chunks: {chunks}")

        if not supports:
            return text, citation_links

        # Sort supports by end_index in descending order to avoid shifting issues when inserting.
        # Ensure segment and end_index exist before accessing and handle None for sorting.
        sorted_supports = sorted(
            [s for s in supports if s.segment and s.segment.end_index is not None],
            key=lambda s: s.segment.end_index if s.segment and s.segment.end_index is not None else -1, # Handle None for sorting
            reverse=True
        )

        # New list to store citation dictionaries
        citation_dicts = []

        for support in sorted_supports:
            # Ensure support.segment is not None before accessing end_index
            if support.segment is None or support.segment.end_index is None:
                continue # Skip this support if segment or end_index is missing

            end_index = support.segment.end_index
            if support.grounding_chunk_indices and chunks: # Check if chunks is not None
                # Create citation string like [1](link1)[2](link2)
                citation_links_for_support = []
                for i in support.grounding_chunk_indices:
                    # Remove previous logging
                    # self.logger.info(f"add_citations: Processing chunk index: {i}")
                    if i < len(chunks):
                        chunk_item = chunks[i]
                        # self.logger.info(f"add_citations: Type of chunk_item (chunks[{i}]): {type(chunk_item)}")
                        # self.logger.info(f"add_citations: Content of chunk_item (chunks[{i}]): {chunk_item}")

                        if hasattr(chunk_item, 'web') and chunk_item.web is not None:
                            web_item = chunk_item.web
                            # self.logger.info(f"add_citations: Type of web_item (chunks[{i}].web): {type(web_item)}")
                            # self.logger.info(f"add_citations: Content of web_item (chunks[{i}].web): {web_item}")

                            if hasattr(web_item, 'uri') and web_item.uri:
                                uri = web_item.uri
                                citation_links_for_support.append(f"[{i + 1}]({uri})")
                                # Add to citation_dicts as a dictionary
                                citation_dicts.append({
                                    'page': 0, # Placeholder, as web search doesn't have page numbers
                                    'confidence': 1.0, # Assume high confidence for direct web citations
                                    'uri': uri, # Include the URI for direct access
                                    'title': getattr(web_item, 'title', 'Web Source') # Include title if available
                                })
                            else:
                                self.logger.warning(f"add_citations: web_item (chunks[{i}].web) does not have a 'uri' attribute or uri is empty.")
                        else:
                            self.logger.warning(f"add_citations: chunk_item (chunks[{i}]) does not have a 'web' attribute or 'web' is None.")
                    else:
                        self.logger.warning(f"add_citations: Chunk index {i} out of bounds for chunks list of length {len(chunks)}.")
                
                if citation_links_for_support:
                    citation_string = ", ".join(citation_links_for_support)
                    text = text[:end_index] + citation_string + text[end_index:]
                    # citation_links.extend(citation_links_for_support) # No longer needed, using citation_dicts

        return text, citation_dicts # Return the list of dictionaries

    def ask_question(self, question: str, expand_query_flag: bool = False, verify_citations_flag: bool = False, web_search: Optional[bool] = False, short_answer: Optional[bool] = False) -> dict:
        """Main interface for asking questions"""

        # 1. Attempt to retrieve relevant chunks from internal DB
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=5)
        
        # 2. Generate initial answer using retrieved context
        rag_result = self.generate_answer(question, relevant_chunks, max_tokens=100 if short_answer else None)

        # Define phrases indicating an inconclusive answer
        inconclusive_phrases = [
            "não pode ser determinado", "não é possível determinar", "não foi encontrada",
            "não está no contexto", "não consigo responder com base no contexto fornecido",
            "não há informações", "não especificado", "não menciona", "não indica",
            "não fornecido", "não disponível", "não se encontra", "não consta",
            "não é possível inferir", "não é possível concluir", "não é possível afirmar",
            "não é possível verificar", "não é possível identificar", "não é possível estabelecer",
            "não é possível precisar", "não é possível obter", "não é possível encontrar",
            "não é possível extrair", "não é possível deduzir", "não é possível derivar",
            "não há informações suficientes no documento",
            "a resposta não está explicitamente mencionada no contexto",
            "o documento não aborda", "não se encontra no texto", "não consta no documento",
            "não é possível inferir do texto", "não é possível deduzir do contexto",
            "não é possível extrair do documento", "não é possível derivar do texto",
            "não é possível obter do contexto", "não é possível encontrar no documento",
            "não é possível precisar com base no contexto",
            "não é possível estabelecer com base no documento",
            "não é possível verificar no texto", "não é possível identificar no contexto",
            "não é possível concluir com base no documento",
            "não é possível afirmar com base no texto",
            "não é possível responder com base no contexto",
            "não é possível determinar a resposta",
            "não é possível encontrar a informação",
            "não é possível fornecer uma resposta definitiva",
            "não é possível dar uma resposta precisa",
            "não é possível responder de forma conclusiva",
            "não é possível fornecer detalhes", "não é possível especificar",
            "não é possível detalhar", "não é possível descrever",
            "não é possível explicar", "não é possível elucidar",
            "não é possível esclarecer", "não é possível apresentar",
            "não é possível indicar", "não é possível mencionar",
            "não é possível citar", "não é possível referir",
            "não é possível aludir", "não é possível fazer referência",
            "não é possível fazer menção", "não é possível fazer alusão",
            "não é possível fazer citação"
        ]

        unconclusive_answer = False
        if rag_result['status'] == 'success' and rag_result['answer']:
            answer_lower = rag_result['answer'].lower()
            for phrase in inconclusive_phrases:
                if phrase.lower() in answer_lower:
                    unconclusive_answer = True
                    break
        else:
            unconclusive_answer = True # If status is not success or answer is empty, consider it inconclusive

        final_answer = rag_result['answer']
        final_citations = rag_result['sources']
        # self.logger.info(f"ask_question: Initial final_citations (from RAG): {final_citations}, Type: {type(final_citations)}")
        
        if unconclusive_answer or web_search == True:
            self.logger.info("Initial answer is inconclusive or web search requested, attempting grounded answer.")
            grounded_response_obj = self.generate_grounded_answer('gemini-2.5-flash', query=question, context_chunks=None, max_tokens=100 if short_answer else None)
            grounded_answer_text, citation_dicts = self.add_citations(response=grounded_response_obj) # Changed to citation_dicts
            # self.logger.info(f"ask_question: Citation links from add_citations: {citation_dicts}, Type: {type(citation_dicts)}") # Changed to citation_dicts
            
            disclaimer = "**Resposta não encontrada no documento, busca no Google foi utilizada!**\n"
            
            # Combine the ungrounded answer with the grounded answer and disclaimer
            final_answer = f"{final_answer}\n\n---\n\n## {disclaimer}\n---\n\n### Resposta (com busca na web): \n{grounded_answer_text}"
            final_citations.extend(citation_dicts) # Extend with dictionaries
            # self.logger.info(f"ask_question: Final final_citations after extension: {final_citations}, Type: {type(final_citations)}")

        # 3. Calculate confidence for RAG answer
        rag_confidence = self.calculate_confidence_score(
            question,
            rag_result['answer'], # Use the original RAG answer for confidence calculation
            rag_result['sources']
        )

        final_confidence = rag_confidence

        # 4. Optional query expansion and verification
        if expand_query_flag:
            expanded_queries = self.expand_query(question)
            # Could implement logic to try expanded queries here
            
        if verify_citations_flag and final_citations:
            verification = self.verify_citations(final_answer, relevant_chunks)
            final_confidence *= verification.get('verification_score', 1.0)

        return {
            'question': question,
            'answer': final_answer,
            'citations': final_citations,
            'confidence_score': final_confidence,
            'retrieved_chunks': len(relevant_chunks),
            'method': 'vector_db'
        }