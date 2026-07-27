# Agente LuminaStore

Agente de inteligencia artificial basado en RAG (Retrieval-Augmented Generation) que responde preguntas de soporte al cliente sobre las políticas de LuminaStore, un ecommerce ficticio. El agente lee un documento PDF con política de privacidad, política de reembolsos y devoluciones, preguntas frecuentes, guía de envíos y entregas, y términos y condiciones, y responde preguntas basándose solo en ese contenido.

Proyecto desarrollado para el Challenge de Alura "Agente de IA que lee y responde preguntas sobre un documento".

App en vivo: https://luminastore-agente-ia-challenge-alura.streamlit.app/

## Arquitectura de la solución

El flujo que sigue el agente es el siguiente:

1. Se carga el documento `documento_luminastore.pdf` con PyPDFLoader.
2. El texto se divide en fragmentos de 1000 caracteres con 200 de solapamiento, usando RecursiveCharacterTextSplitter.
3. Cada fragmento se convierte en un vector con CohereEmbeddings (modelo embed-multilingual-v3.0).
4. Los vectores se guardan en un índice FAISS para poder buscar por significado, no solo por palabras exactas.
5. Cuando el usuario hace una pregunta, el retriever busca los fragmentos más relacionados con esa pregunta.
6. Esos fragmentos se le pasan como contexto al modelo de lenguaje (ChatCohere, modelo command-a-03-2025), junto con un prompt que le indica responder solo con base en esa información.
7. La respuesta se muestra en una interfaz de chat (Streamlit en producción, Gradio para pruebas locales).

La idea principal detrás de este enfoque (RAG) es que el modelo no responde de memoria ni inventa información: solo usa lo que realmente está en el PDF.

## Tecnologías usadas

- Python
- LangChain / LangChain Classic
- PyPDF
- Cohere (embeddings y modelo de lenguaje)
- FAISS
- Streamlit (interfaz principal, desplegada en producción)
- Gradio (interfaz alternativa para pruebas en Colab)
- Streamlit Community Cloud (hosting)

## Cómo ejecutar el proyecto

Lo más rápido es entrar directo a la app ya desplegada:
https://luminastore-agente-ia-challenge-alura.streamlit.app/

Si quieres correrlo en tu máquina:

```
git clone https://github.com/Cristianriosrivas/luminastore-agente-ia.git
cd luminastore-agente-ia
pip install -r requirements.txt
cp .env.example .env
```

Edita el archivo `.env` y pega ahí tu propia API key de Cohere. Luego corre:

```
streamlit run streamlit_app.py
```

También existe `app.py`, una versión con Gradio pensada para correr y probar dentro de Google Colab.

## Ejemplos de preguntas y respuestas

Pregunta: ¿Cuánto cuesta el envío estándar y qué pasa si mi código postal es de difícil acceso?

Respuesta del agente: El costo del envío estándar es de $6.50 USD y tiene un tiempo de tránsito de 4 a 7 días hábiles con cobertura nacional. Si el código postal es considerado "Zona Extendida" (áreas rurales, fronterizas o de difícil acceso logístico), se aplica un recargo fijo de $5.00 USD adicional y el tiempo de entrega puede extenderse entre 3 y 5 días hábiles adicionales.

Pregunta: ¿Cuál es la política de reembolsos?

Respuesta del agente: (pegar aquí la respuesta que dio el agente)

Pregunta: ¿Cómo protegen mis datos personales?

Respuesta del agente: (pegar aquí la respuesta que dio el agente)

Pregunta: ¿Cuáles son los términos y condiciones para devolver un producto?

Respuesta del agente: (pegar aquí la respuesta que dio el agente)

## Sobre el despliegue

El agente está desplegado en Streamlit Community Cloud:
https://luminastore-agente-ia-challenge-alura.streamlit.app/

(agregar aquí una captura de pantalla de la app funcionando)

El plan original era desplegar en Oracle Cloud Infrastructure (OCI), pero durante el desarrollo la región de Bogotá tuvo una indisponibilidad prolongada de capacidad para instancias gratuitas (error "Out of host capacity", un problema reconocido por la propia Oracle). Por eso terminé usando Streamlit Community Cloud, que es igual de gratuito y se ajustaba mejor al tiempo que tenía disponible.

## Estructura del repositorio

- `streamlit_app.py`: script principal, agente con interfaz Streamlit, es lo que corre en el deploy.
- `app.py`: versión con interfaz Gradio, para pruebas en Google Colab.
- `requirements.txt`: dependencias del proyecto.
- `.env.example`: plantilla de variables de entorno.
- `.gitignore`: archivos que Git no debe subir (incluye `.env`).
- `documento_luminastore.pdf`: documento fuente que usa el agente.
- `README.md`: este archivo.
