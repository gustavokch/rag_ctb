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
    # Placeholder for a PDF file. In a real scenario, you'd have a PDF here.
    # For testing purposes, you might want to create a dummy PDF or skip this part
    # if you're only testing the Streamlit interface's rendering.
    try:
        rag_system = setup_legislation_rag_system("ctb.pdf")
        time.sleep(5)  # Allow time for the interface to load
        create_web_interface(rag_system)
    except FileNotFoundError:
        st.error("Por favor, certifique-se de que 'ctb.pdf' exista no mesmo diretório.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a configuração ou execução do sistema: {e}")
        