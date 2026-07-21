"""
Agente LuminaStore - Asistente de soporte al cliente basado en RAG (Retrieval-Augmented Generation).
Lee un PDF con las políticas de la tienda y responde preguntas sobre su contenido.
"""

import os
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. Configuración: leemos la API key y la ruta del PDF desde variables de entorno
# ---------------------------------------------------------------------------
load_dotenv()  # carga el archivo .env si existe (solo para desarrollo local)

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PDF_PATH = os.environ.get("PDF_PATH", "documento_luminastore.pdf")

if not COHERE_API_KEY:
    raise RuntimeError(
        "Falta la variable de entorno COHERE_API_KEY. "
        "Crea un archivo .env (basado en .env.example) o expórtala en tu terminal:\n"
        "  export COHERE_API_KEY=tu_key_aqui"
    )

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo '{PDF_PATH}'. "
        "Verifica que el PDF esté en la misma carpeta que app.py, "
        "o ajusta la variable PDF_PATH en tu .env."
    )

os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# ---------------------------------------------------------------------------
# 2. Cargamos y troceamos el documento
# ---------------------------------------------------------------------------
print(f"1/4 - Leyendo y preparando '{PDF_PATH}'...")
loader = PyPDFLoader(PDF_PATH)
paginas = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
fragmentos = text_splitter.split_documents(paginas)
print(f"   -> {len(fragmentos)} fragmentos generados.")

# ---------------------------------------------------------------------------
# 3. Creamos los embeddings, el índice vectorial y el modelo
# ---------------------------------------------------------------------------
print("2/4 - Generando embeddings y creando el índice FAISS...")
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
vectorstore = FAISS.from_documents(fragmentos, embeddings)
retriever = vectorstore.as_retriever()

print("3/4 - Conectando con el modelo de Cohere...")
llm = ChatCohere(model="command-a-03-2025")

# ---------------------------------------------------------------------------
# 4. Armamos la cadena de preguntas y respuestas
# ---------------------------------------------------------------------------
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
agente_luminastore = create_retrieval_chain(retriever, question_answer_chain)

print("4/4 - ¡Agente listo!\n")


# ---------------------------------------------------------------------------
# 5. Función que conecta el agente con la interfaz Gradio
# ---------------------------------------------------------------------------
def responder(pregunta, historial):
    """
    Gradio le pasa a esta función el mensaje nuevo y el historial del chat.
    Nosotros solo necesitamos el mensaje nuevo para consultar al agente.
    """
    try:
        resultado = agente_luminastore.invoke({"input": pregunta})
        return resultado["answer"]
    except Exception as e:
        return f"Ocurrió un error al procesar tu pregunta: {e}"


# ---------------------------------------------------------------------------
# 6. Interfaz Gradio
# ---------------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=responder,
    title="🛍️ Agente LuminaStore",
    description=(
        "Pregúntame sobre política de privacidad, reembolsos y devoluciones, "
        "envíos y entregas, términos y condiciones, o preguntas frecuentes de LuminaStore."
    ),
    examples=[
        "¿Cuánto cuesta el envío estándar?",
        "¿Cuál es la política de reembolsos?",
        "¿Cómo protegen mis datos personales?",
        "¿Qué pasa si mi código postal es de difícil acceso?",
    ],
)

if __name__ == "__main__":
    # server_name="0.0.0.0" es clave para que, cuando esto corra en una VM de OCI,
    # sea accesible desde fuera de la máquina (no solo desde localhost).
    demo.launch(server_name="0.0.0.0", server_port=7860)
