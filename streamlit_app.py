import time
import streamlit as st

from system_setup import setup_legislation_rag_system
from rag_pipeline import ParsedQuestion # Import the dataclass

def create_web_interface(rag_system):
    st.title("📚 Assistente de Estudo de Legislação")
    
    tab1, tab2 = st.tabs(["Pergunta Direta", "Banco de Questões"])

    with tab1:
        st.header("Pergunta Direta")
        # Question input
        question = st.text_area("Faça uma pergunta sobre a legislação:")
        allow_web_search = st.checkbox("Permitir pesquisa na web", key="tab1_web_search")
        short_answer = st.checkbox("Resposta curta", key="tab1_short_answer")
        if question and st.button("Pesquisar", key="tab1_search"):
            with st.spinner("Pesquisando e gerando resposta..."):
                if allow_web_search:
                    # If web search is allowed, use the rag system with web search enabled
                    result = rag_system.ask_question(question, web_search=True)
                else:
                    result = rag_system.ask_question(question)
                
                # Display answer
                st.subheader("Resposta")
                if result['answer']:
                    st.write(result['answer'])
                
                # Display confidence
                confidence = result['confidence_score']
                st.metric("Pontuação de Confiança", f"{confidence:.2%}")
                
                # Display citations
                st.subheader("Citações")
                for cite in result['citations']:
                    if 'page' in cite and 'confidence' in cite: # Check for RAG citations
                        st.write(f"📄 Página {cite['page']} (confiança: {cite['confidence']:.2%})")
                    elif 'uri' in cite: # Check for web citations
                        st.write(f"🔗 Fonte: [{cite.get('title', 'Web Source')}]({cite['uri']})")


    with tab2:
        st.header("Banco de Questões")
        uploaded_file = st.file_uploader("Carregue um arquivo de texto com o banco de questões", type="txt")

        if uploaded_file is not None:
            try:
                # To read file as string:
                file_content = str(uploaded_file.read(), "utf-8")
                parsed_questions = rag_system.parse_question_bank(file_content)

                if not parsed_questions:
                    st.warning("Nenhuma questão foi encontrada ou o formato do arquivo é inválido.")
                else:
                    st.success(f"{len(parsed_questions)} questões carregadas com sucesso!")
                    
                    # Pagination
                    questions_per_page = 10
                    total_pages = (len(parsed_questions) + questions_per_page - 1) // questions_per_page
                    
                    if 'current_page_qb' not in st.session_state:
                        st.session_state['current_page_qb'] = 1
                    
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        if st.button("Anterior", disabled=st.session_state['current_page_qb'] == 1):
                            st.session_state['current_page_qb'] -= 1
                    with col2:
                        st.write(f"Página {st.session_state['current_page_qb']} de {total_pages}")
                    with col3:
                        if st.button("Próxima", disabled=st.session_state['current_page_qb'] == total_pages):
                            st.session_state['current_page_qb'] += 1
                    
                    start_index = (st.session_state['current_page_qb'] - 1) * questions_per_page
                    end_index = start_index + questions_per_page
                    current_questions = parsed_questions[start_index:end_index]

                    for i, pq in enumerate(current_questions):
                        st.markdown(f"---\n### Questão {pq.number}")
                        st.write(pq.text)
                        for key, value in pq.options.items():
                            st.write(f"**{key})** {value}")
                        
                        # Use a unique key for each button based on question number and page
                        if st.button(f"Gerar Resposta para Questão {pq.number}", key=f"gen_ans_{pq.number}_p{st.session_state['current_page_qb']}"):
                            # For generate_answer, we need to pass the full question text and the options as part of the query
                            # to ensure the model can choose the correct one.
                            # The generate_answer method itself is designed to handle multiple choice.
                            query_for_generation = pq.full_question_text
                            
                            with st.spinner(f"Gerando resposta para a Questão {pq.number}..."):
                                # We use generate_answer directly as we want the RAG response based on the document,
                                # not the full ask_question logic which includes web search fallback by default.
                                # If web search is desired for this tab too, it would need to be added.
                                relevant_chunks = rag_system.retrieve_relevant_chunks(query_for_generation, top_k=5)
                                answer_result = rag_system.generate_grounded_answer(model = 'gemini-2.5-flash',query=query_for_generation, context_chunks=relevant_chunks, max_tokens=None)

                                st.subheader(f"Resposta Gerada para Questão {pq.number}")
                                if answer_result['status'] == 'success' and answer_result['answer']:
                                    st.write(answer_result['answer'])
                                    
                                    # Display confidence for this specific answer
                                    if 'confidence_score' not in answer_result: # generate_answer doesn't calculate it directly
                                        confidence = rag_system.calculate_confidence_score(
                                            query_for_generation,
                                            answer_result['answer'],
                                            answer_result['sources']
                                        )
                                        st.metric("Pontuação de Confiança", f"{confidence:.2%}")
                                    else:
                                         st.metric("Pontuação de Confiança", f"{answer_result['confidence_score']:.2%}")


                                    st.subheader("Citações da Resposta")
                                    for cite in answer_result['sources']:
                                        st.write(f"📄 Página {cite['page']} (confiança: {cite['confidence']:.2%})")
                                else:
                                    st.error(f"Não foi possível gerar uma resposta para a Questão {pq.number}. Erro: {answer_result.get('answer', 'Desconhecido')}")

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
                st.exception(e)

if __name__ == "__main__":
    # Placeholder for a PDF file. In a real scenario, you'd have a PDF here.
    # For testing purposes, you might want to create a dummy PDF or skip this part
    # if you're only testing the Streamlit interface's rendering.
    try:
        rag_system = setup_legislation_rag_system(pdf_path="ctb.pdf")
        time.sleep(5)  # Allow time for the interface to load
        create_web_interface(rag_system)
    except FileNotFoundError:
        st.error("Por favor, certifique-se de que 'ctb.pdf' exista no mesmo diretório.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a configuração ou execução do sistema: {e}")
        