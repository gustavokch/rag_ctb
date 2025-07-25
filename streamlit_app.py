import time
import streamlit as st

from system_setup import setup_legislation_rag_system
def create_web_interface(rag_system):
    st.title("📚 Assistente de Estudo de Legislação")
    
    # Question input
    question = st.text_area("Faça uma pergunta sobre a legislação:")
    
    if question and st.button("Pesquisar"):
        with st.spinner("Pesquisando e gerando resposta..."):
            result = rag_system.ask_question(question)
            
            # Display answer
            st.subheader("Resposta")
            st.write(result['answer'])
            
            # Display confidence
            confidence = result['confidence_score']
            st.metric("Pontuação de Confiança", f"{confidence:.2%}")
            
            # Display citations
            st.subheader("Citações")
            for cite in result['citations']:
                st.write(f"📄 Página {cite['page']} (confiança: {cite['confidence']:.2%})")

if __name__ == "__main__":
    st.title("Configuração do Sistema RAG de Legislação")
    pdf_path = st.text_input("Por favor, insira o caminho do arquivo PDF da legislação (ex: ctb.pdf):", "ctb.pdf")

    if st.button("Carregar Legislação"):
        if pdf_path:
            with st.spinner(f"Carregando legislação de '{pdf_path}'..."):
                try:
                    rag_system = setup_legislation_rag_system(pdf_path)
                    st.success("Legislação carregada com sucesso!")
                    time.sleep(1) # Small delay for message to be seen
                    create_web_interface(rag_system)
                except FileNotFoundError:
                    st.error(f"Erro: O arquivo '{pdf_path}' não foi encontrado. Por favor, verifique o caminho.")
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a configuração ou execução do sistema: {e}")
        else:
            st.warning("Por favor, insira um caminho de arquivo PDF.")
        