import subprocess
import sys
import time
import streamlit as st
from system_setup import setup_legislation_rag_system # Assuming system_setup.py is in the same directory

def create_web_interface(rag_system):
    st.title("📚 Legislation Study Assistant")
    
    # Question input
    question = st.text_input("Ask a question about the legislation:")
    
    if question and st.button("Search"):
        with st.spinner("Searching and generating answer..."):
            result = rag_system.ask_question(question)
            
            # Display answer
            st.subheader("Answer")
            st.write(result['answer'])
            
            # Display confidence
            confidence = result['confidence_score']
            st.metric("Confidence Score", f"{confidence:.2%}")
            
            # Display citations
            st.subheader("Citations")
            for cite in result['citations']:
                st.write(f"📄 Page {cite['page']} (confidence: {cite['confidence']:.2%})")

def deploy_streamlit_with_cloudflared(script_path: str, port: int = 35333):
    """Deploys the Streamlit app via a temporary Cloudflare tunnel on the specified port."""
    # Ensure cloudflared is installed
    try:
        import cloudflared
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudflared"])

    # Start Streamlit app
    streamlit_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", script_path, "--server.port", str(port)
    ])
    time.sleep(5)  # Wait for Streamlit to start
    # Start Cloudflare tunnel
    tunnel_proc = subprocess.Popen([
        "cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(f"🌐 Streamlit running on port {port}. Spinning up Cloudflare tunnel...")
    public_url = None
    # Parse tunnel output for the public URL
    if tunnel_proc.stdout is not None:
        for line in iter(tunnel_proc.stdout.readline, ""):
            if not "Requesting" in line and "trycloudflare.com" in line:
                public_url = line.strip().split()
                print(f"🔗 Public URL: {public_url}")
                break
    if not public_url:
        print("❌ Could not retrieve Cloudflare tunnel URL.")
    print("Press Ctrl+C to stop the server and tunnel.")
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        streamlit_proc.terminate()
        tunnel_proc.terminate()

if __name__ == "__main__":
    # Placeholder for a PDF file. In a real scenario, you'd have a PDF here.
    # For testing purposes, you might want to create a dummy PDF or skip this part
    # if you're only testing the Streamlit interface's rendering.
    try:
        rag_system = setup_legislation_rag_system("ctb.pdf")
        create_web_interface(rag_system)
        time.sleep(5)  # Allow time for the interface to load
        deploy_streamlit_with_cloudflared("web_interface.py", port=35333)
    except FileNotFoundError:
        st.error("Please ensure 'ctb.pdf' exists in the same directory.")
    except Exception as e:
        st.error(f"An error occurred during system setup or execution: {e}")
        