"""
Agente LuminaStore - Asistente de soporte al cliente basado en RAG (Retrieval-Augmented Generation).
Lee un PDF con las políticas de la tienda y responde preguntas sobre su contenido,
en una interfaz con apariencia de tienda online.
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
# 1. Configuración: leemos la API key y la ruta del PDF desde variables de entorno.
#    En local, vienen de un archivo .env. En Hugging Face Spaces, vienen de los
#    "Secrets" configurados en el panel del Space - en ningún caso hay que
#    escribir la key a mano dentro del código.
# ---------------------------------------------------------------------------
load_dotenv()  # carga el archivo .env si existe (solo aplica en desarrollo local)

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PDF_PATH = os.environ.get("PDF_PATH", "documento_luminastore.pdf")

if not COHERE_API_KEY:
    raise RuntimeError(
        "Falta la variable de entorno COHERE_API_KEY.\n"
        "- En local: crea un archivo .env (basado en .env.example) con tu key.\n"
        "- En Hugging Face Spaces: ve a Settings -> Variables and secrets -> "
        "New secret, y agrega COHERE_API_KEY con tu valor."
    )

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo '{PDF_PATH}'. "
        "Verifica que el PDF esté en la misma carpeta que app.py, "
        "o ajusta la variable PDF_PATH."
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
# 5. Lógica del chat
# ---------------------------------------------------------------------------
def responder(mensaje, historial):
    """Recibe la pregunta y el historial (lista de dicts role/content de Gradio),
    consulta al agente y devuelve el historial actualizado + limpia el textbox."""
    historial = historial or []
    if not mensaje or not mensaje.strip():
        return historial, ""

    try:
        resultado = agente_luminastore.invoke({"input": mensaje})
        respuesta = resultado["answer"]
    except Exception as e:
        respuesta = f"Ocurrió un error al procesar tu pregunta: {e}"

    historial = historial + [
        {"role": "user", "content": mensaje},
        {"role": "assistant", "content": respuesta},
    ]
    return historial, ""


def enviar_ejemplo(pregunta, historial):
    return responder(pregunta, historial)


# ---------------------------------------------------------------------------
# 6. Estilos - apariencia de tienda online para LuminaStore
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --ls-primary: #4c1d95;
    --ls-primary-dark: #2e1065;
    --ls-gold: #eab308;
    --ls-bg: #faf8f5;
}
.gradio-container {
    background: var(--ls-bg) !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
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
    font-size: 28px;
    color: white;
}
#ls-header h1 span {
    color: var(--ls-gold);
}
#ls-header p {
    margin: 0;
    opacity: 0.9;
    font-size: 15px;
}
#ls-sidebar {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #ece7f5;
}
#ls-sidebar h3 {
    margin-top: 0;
    color: var(--ls-primary-dark);
}
#ls-chatbot {
    border-radius: 14px !important;
    border: 1px solid #ece7f5 !important;
}
#ls-send-btn {
    background: var(--ls-primary) !important;
    border: none !important;
}
.ls-example-btn {
    text-align: left !important;
}
"""

EJEMPLOS = [
    "¿Cuánto cuesta el envío estándar?",
    "¿Cuál es la política de reembolsos?",
    "¿Cómo protegen mis datos personales?",
    "¿Qué pasa si mi código postal es de difícil acceso?",
]

# ---------------------------------------------------------------------------
# 7. Interfaz
# ---------------------------------------------------------------------------
with gr.Blocks(title="LuminaStore | Asistente Virtual") as demo:
    gr.HTML(
        """
        <div id="ls-header">
            <h1>✨ <span>Lumina</span>Store</h1>
            <p>Tu asistente virtual de soporte al cliente, disponible 24/7</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="ls-sidebar"):
            gr.Markdown(
                "### 🛍️ Puedo ayudarte con\n"
                "- 🔒 Política de privacidad\n"
                "- 💰 Reembolsos y devoluciones\n"
                "- 🚚 Envíos y entregas\n"
                "- 📜 Términos y condiciones\n"
                "- ❓ Preguntas frecuentes"
            )
            gr.Markdown("### 💬 Preguntas rápidas")
            botones_ejemplo = [
                gr.Button(ej, elem_classes="ls-example-btn", size="sm")
                for ej in EJEMPLOS
            ]

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=460,
                elem_id="ls-chatbot",
                label=None,
                show_label=False,
                placeholder="Escríbeme algo como: '¿Cuál es la política de reembolsos?'",
            )
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Escribe tu pregunta aquí...",
                    scale=5,
                    container=False,
                )
                enviar_btn = gr.Button(
                    "Enviar", scale=1, variant="primary", elem_id="ls-send-btn"
                )

    enviar_btn.click(responder, [txt, chatbot], [chatbot, txt])
    txt.submit(responder, [txt, chatbot], [chatbot, txt])

    for boton, pregunta in zip(botones_ejemplo, EJEMPLOS):
        boton.click(
            lambda historial, p=pregunta: enviar_ejemplo(p, historial),
            [chatbot],
            [chatbot, txt],
        )

if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        # Estamos corriendo dentro de Hugging Face Spaces
        demo.launch(css=CUSTOM_CSS)
    else:
        # Estamos corriendo localmente
        demo.launch(css=CUSTOM_CSS, server_name="0.0.0.0", server_port=7860)
