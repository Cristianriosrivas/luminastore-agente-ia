"""
Agente LuminaStore - Asistente de soporte al cliente basado en RAG (Retrieval-Augmented Generation).
Versión con Streamlit, pensada para desplegarse en Streamlit Community Cloud.
"""

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LuminaStore | Asistente Virtual",
    page_icon="✨",
    layout="wide",
)

PDF_PATH = os.environ.get("PDF_PATH", "documento_luminastore.pdf")


def obtener_api_key():
    """
    Busca la API key de Cohere en este orden:
    1. Variables de entorno (útil en local, vía .env)
    2. st.secrets (así es como Streamlit Community Cloud guarda los secretos:
       se configura UNA sola vez en el panel del sitio, y desde ahí el agente
       la lee automáticamente en cada inicio, sin volver a escribirla nunca).
    """
    load_dotenv()
    key = os.environ.get("COHERE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["COHERE_API_KEY"]
    except Exception:
        return None


COHERE_API_KEY = obtener_api_key()

if not COHERE_API_KEY:
    st.error(
        "⚠️ Falta configurar la API key de Cohere.\n\n"
        "- En local: crea un archivo `.env` (basado en `.env.example`) con tu key.\n"
        "- En Streamlit Community Cloud: ve a la configuración de tu app → "
        "**Settings → Secrets**, y agrega:\n\n"
        "```\nCOHERE_API_KEY = \"tu_key_aqui\"\n```"
    )
    st.stop()

os.environ["COHERE_API_KEY"] = COHERE_API_KEY


# ---------------------------------------------------------------------------
# 2. Cargamos el agente (con cache para no rehacer todo en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparando el agente LuminaStore...")
def cargar_agente():
    if not os.path.exists(PDF_PATH):
        st.error(f"No se encontró el archivo '{PDF_PATH}'.")
        st.stop()

    loader = PyPDFLoader(PDF_PATH)
    paginas = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(paginas)

    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = FAISS.from_documents(fragmentos, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatCohere(model="command-a-03-2025")

    instrucciones = (
        "Eres un asistente virtual experto en soporte al cliente para LuminaStore. "
        "Usa EXCLUSIVAMENTE los siguientes fragmentos de información para responder. "
        "Si la respuesta no está en el texto, di amablemente que no tienes esa información. "
        "Responde de forma profesional, clara y directa.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", instrucciones),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


agente_luminastore = cargar_agente()


# ---------------------------------------------------------------------------
# 3. Estilos - apariencia de tienda online para LuminaStore
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ls-primary: #4c1d95;
        --ls-primary-dark: #2e1065;
        --ls-gold: #eab308;
    }
    #ls-header {
        background: linear-gradient(120deg, var(--ls-primary-dark), var(--ls-primary));
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 18px;
        color: white;
        text-align: center;
    }
    #ls-header h1 {
        margin: 0 0 6px 0;
        color: white;
    }
    #ls-header h1 span {
        color: var(--ls-gold);
    }
    #ls-header p {
        margin: 0;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div id="ls-header">
        <h1>✨ <span>Lumina</span>Store</h1>
        <p>Tu asistente virtual de soporte al cliente, disponible 24/7</p>
    </div>
    """,
    unsafe_allow_html=True,
)

EJEMPLOS = [
    "¿Cuánto cuesta el envío estándar?",
    "¿Cuál es la política de reembolsos?",
    "¿Cómo protegen mis datos personales?",
    "¿Qué pasa si mi código postal es de difícil acceso?",
]

# ---------------------------------------------------------------------------
# 4. Panel lateral: categorías de ayuda y preguntas rápidas
# ---------------------------------------------------------------------------
pregunta_desde_boton = None
with st.sidebar:
    st.markdown("### 🛍️ Puedo ayudarte con")
    st.markdown(
        "- 🔒 Política de privacidad\n"
        "- 💰 Reembolsos y devoluciones\n"
        "- 🚚 Envíos y entregas\n"
        "- 📜 Términos y condiciones\n"
        "- ❓ Preguntas frecuentes"
    )
    st.markdown("### 💬 Preguntas rápidas")
    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, use_container_width=True):
            pregunta_desde_boton = ejemplo

# ---------------------------------------------------------------------------
# 5. Chat
# ---------------------------------------------------------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

pregunta_escrita = st.chat_input("Escribe tu pregunta aquí...")
pregunta_final = pregunta_desde_boton or pregunta_escrita

if pregunta_final:
    st.session_state.mensajes.append({"role": "user", "content": pregunta_final})
    with st.chat_message("user"):
        st.markdown(pregunta_final)

    with st.chat_message("assistant"):
        with st.spinner("Consultando las políticas de LuminaStore..."):
            try:
                resultado = agente_luminastore.invoke({"input": pregunta_final})
                respuesta = resultado["answer"]
            except Exception as e:
                respuesta = f"Ocurrió un error al procesar tu pregunta: {e}"
        st.markdown(respuesta)

    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
