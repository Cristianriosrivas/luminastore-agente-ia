# luminastore-agente-ia
Agente de inteligencia artificial basado en RAG (Retrieval-Augmented Generation) que responde preguntas de soporte al cliente sobre las políticas de LuminaStore (un ecommerce), a partir de un documento PDF con política de privacidad, política de reembolsos y devoluciones, preguntas frecuentes, guía de envíos y entregas, y términos y condiciones.

Proyecto desarrollado como parte del Challenge de Alura: Agente de IA que lee y responde preguntas sobre un documento.


## Arquitectura de la solución

El agente sigue un flujo clásico de RAG (Retrieval-Augmented Generation):

documento_luminastore.pdf
        │
        ▼
  [PyPDFLoader]  →  carga el PDF y extrae su texto
        │
        ▼
  [RecursiveCharacterTextSplitter]  →  divide el texto en fragmentos de 1000
        │                                caracteres (200 de solapamiento)
        ▼
  [CohereEmbeddings]  →  convierte cada fragmento en un vector numérico
        │
        ▼
  [FAISS Vectorstore]  →  índice vectorial en memoria para búsqueda semántica
        │
        ▼
  [Retriever]  →  dada una pregunta, busca los fragmentos más relevantes
        │
        ▼
  [ChatCohere + Prompt]  →  el LLM redacta la respuesta usando SOLO esos
        │                     fragmentos como contexto
        ▼
  [Gradio ChatInterface]  →  interfaz de chat donde el usuario pregunta
                              y recibe la respuesta

En resumen: cuando alguien hace una pregunta, el sistema busca los fragmentos del PDF más relacionados semánticamente con esa pregunta (no es una búsqueda de palabras exactas, sino de significado), y se los pasa al modelo de lenguaje para que redacte una respuesta basada únicamente en esa información — evitando que el modelo invente datos que no están en el documento.

## Tecnologías utilizadas

| Tecnología | Uso |
| Python | Lenguaje principal |
| LangChain / LangChain Classic | Orquestación del pipeline RAG |
| PyPDF | Lectura del documento PDF |
| Cohere (`embed-multilingual-v3.0`) | Generación de embeddings |
| Cohere (`command-a-03-2025`) | Modelo de lenguaje (LLM) |
| FAISS | Base de datos vectorial (búsqueda semántica) |
| Gradio | Interfaz de chat |
| Oracle Cloud Infrastructure (OCI) | Despliegue en la nube |


## Cómo ejecutar el proyecto

### Opción A: Google Colab (recomendado para probar rápido)

1. Sube `documento_luminastore` a la sesión de Colab.
2. Ejecuta las celdas del notebook en orden.
3. Cuando te pida la API Key de Cohere, pégala (puedes conseguir una gratis en [dashboard.cohere.com](https://dashboard.cohere.com)).
4. Al final aparecerá una interfaz de chat embebida en el notebook.

### Opción B: localmente con `app.py`

bash
[git clone https://github.com/TU_USUARIO/luminastore-agente-ia.git
cd luminastore-agente-ia](https://github.com/Cristianriosrivas/luminastore-agente-ia.git)](https://github.com/Cristianriosrivas/luminastore-agente-ia.git)
pip install -r requirements.txt
cp .env.example .env
# Edita .env y pega tu COHERE_API_KEY
python app.py

Luego abre en tu navegador la URL que aparece en la terminal (por defecto `http://localhost:7860`).

## Ejemplos de preguntas y respuestas

Pregunta: ¿Cuánto cuesta el envío estándar y qué pasa si mi código postal es de difícil acceso?

Respuesta del agente: El costo del envío estándar es de $6.50 USD y tiene un tiempo de tránsito de 4 a 7 días hábiles con cobertura nacional. Si el código postal es considerado "Zona Extendida" (áreas rurales, fronterizas o de difícil acceso logístico), se aplica un recargo fijo de $5.00 USD adicional y el tiempo de entrega puede extenderse entre 3 y 5 días hábiles adicionales.

Pregunta: ¿Cuál es la política de reembolsos?

Respuesta del agente: La política de reembolsos de LuminaStore tiene como objetivo garantizar la satisfacción total de los clientes. Los reembolsos se realizan al mismo método de pago original y no se hacen en efectivo ni a cuentas de terceros. Los tiempos de acreditación varían según el método de pago: de 5 a 10 días hábiles para tarjetas de crédito/débito.

Los plazos para solicitar una devolución dependen del motivo, y hay excepciones para artículos no retornables, como tarjetas de regalo, software, productos de higiene abiertos, artículos personalizados y productos en liquidación final, salvo por defectos comprobados.

Si la devolución es a cargo del cliente (por retracto, error de compra o incompatibilidad), se descuenta una tarifa fija de $8.50 USD por logística inversa.

Pregunta: ¿Cómo protegen mis datos personales?

Respuesta del agente: En LuminaStore, protegemos sus datos personales implementando medidas de seguridad técnicas, administrativas y físicas de vanguardia. Utilizamos tecnología de encriptación SSL (Secure Socket Layer) de 256 bits para proteger los datos durante su transmisión en internet. Además, contamos con firewalls avanzados y controles de acceso restringido en nuestras bases de datos. Estas medidas están diseñadas para garantizar la confidencialidad, integridad y disponibilidad de su información personal.

Pregunta:¿Cuáles son los términos y condiciones para devolver un producto?

Respuesta del agente: Para que una solicitud de devolución sea aprobada en LuminaStore, el producto debe cumplir con los siguientes requisitos:

Estado Físico: El artículo debe estar sin uso, en perfectas condiciones estéticas y operativas.
Embalaje: Debe devolverse en su caja o empaque original, sin alteraciones, rayones o roturas significativas.
Accesorios Completos: Debe incluir todos los manuales, cables, piezas adicionales y obsequios promocionales que hayan venido en la caja original.
Comprobante: Es indispensable adjuntar la factura de compra, recibo electrónico o el número de orden oficial (Order ID).
Además, es importante considerar los plazos para solicitar una devolución:

Derecho de retracto (arrepentimiento): 30 días naturales desde la fecha de entrega.
Producto dañado en tránsito: 48 horas posteriores a la entrega.
Producto incorrecto: 5 días hábiles desde la recepción.
Defectos de fábrica: Cubierto por la garantía (12 meses para electrónicos y 3 meses para accesorios).
Tenga en cuenta que existen excepciones para artículos no retornables, como tarjetas de regalo, software, productos de cuidado personal abiertos, artículos personalizados y productos en liquidación final.

## ☁️ Evidencia del despliegue en OCI



## Estructura del repositorio

luminastore-agente-ia/
├── app.py                     # Script principal (agente + interfaz Gradio)
├── requirements.txt           # Dependencias del proyecto
├── .env.example                # Plantilla de variables de entorno
├── .gitignore                  # Archivos excluidos de Git (incluye .env)
├── documento_luminastore.pdf  # Documento fuente del agente
└── README.md                   # Este archivo
