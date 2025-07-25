import subprocess
import sys
import time
import streamlit as st
from system_setup import setup_legislation_rag_system # Assuming system_setup.py is in the same directory

def deploy_streamlit_with_cloudflared(script_path: str, port: int = 35333):
    """Implanta o aplicativo Streamlit através de um túnel temporário do Cloudflare na porta especificada."""
    # Ensure cloudflared is installed
    try:
        import cloudflared
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudflared"])

    # Start Streamlit app
    streamlit_proc = subprocess.Popen([
        "/home/ubuntu/Git/rag_ctb/.venv/bin/streamlit", "run", script_path, "--server.port", str(port)
    ])
    time.sleep(5)  # Wait for Streamlit to start
    # Start Cloudflare tunnel
    tunnel_proc = subprocess.Popen([
        "cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(f"🌐 Streamlit rodando na porta {port}. Iniciando túnel Cloudflare...")
    public_url = None
    # Parse tunnel output for the public URL
    if tunnel_proc.stdout is not None:
        for line in iter(tunnel_proc.stdout.readline, ""):
            if not "Requesting" in line and "trycloudflare.com" in line:
                public_url = line.strip().split("https://")[1]
                public_url.split("|")[-1]  # This line is not a string to be translated, it's code.
                print(f"🔗 URL Pública: https://{public_url}")
                break
    if not public_url:
        print("❌ Não foi possível obter a URL do túnel Cloudflare.")
    print("Pressione Ctrl+C para parar o servidor e o túnel.")
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
        time.sleep(5)  # Allow time for the interface to load
        deploy_streamlit_with_cloudflared("streamlit_app.py", port=35333)
    except FileNotFoundError:
        st.error("Por favor, certifique-se de que 'ctb.pdf' exista no mesmo diretório.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a configuração ou execução do sistema: {e}")
        