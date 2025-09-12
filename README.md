# Proyecto Práctico 1: Asistente CHATBOT sobre normativa y reglamentos de la UFRO

## 🚀 Descripción del Proyecto

Este proyecto consiste en el desarrollo de un **asistente conversacional RAG (Retrieval-Augmented Generation)** en Python. El objetivo principal es responder preguntas de estudiantes y personal sobre la normativa y reglamentos de la Universidad de La Frontera (UFRO), citando las fuentes oficiales.

El asistente integra dos proveedores de modelos de lenguaje grandes (LLM): **ChatGPT API** y **DeepSeek API**, mediante un **patrón de proveedor** que permite la comparación (A/B) o el consenso (ensemble) de sus respuestas.

### Tecnologías Mínimas Requeridas
* **Python 3.11+**
* `openai` SDK
* **DeepSeek API** (compatible con OpenAI)
* **RAG** con **FAISS**
* **`sentence-transformers`**
* **`pypdf`**

---

## 🎯 Objetivos de Aprendizaje

A través de este proyecto, se busca:

* Integrar **dos proveedores LLM** (ChatGPT y DeepSeek) usando un **patrón de proveedor** en Python.
* Construir un **asistente RAG** que responda preguntas sobre **normativa UFRO** (reglamentos, calendarios, procedimientos), citando las fuentes.
* Diseñar **prompts robustos** (rol del sistema, políticas de cita, manejo de abstención).
* Implementar la **ingesta y vectorización de PDFs** con FAISS y `sentence-transformers`.
* Medir la **calidad** (precisión/cobertura), **latencia** y **costo por consulta**, comparando el rendimiento de ChatGPT vs DeepSeek.
* Aplicar **consideraciones éticas** (privacidad, atribución, sesgos, transparencia).

---

## 📄 Enunciado Detallado

El asistente conversacional deberá:

1.  **Ingesta de Documentos**: Procesar un conjunto de documentos oficiales (PDF/HTML) de normativa UFRO. Se debe registrar la procedencia (URL/fecha de descarga) y la fecha de vigencia de cada documento.
2.  **Índice Vectorial**: Construir un **índice vectorial (FAISS)**, fragmentando los documentos en *chunks* con solapamiento.
3.  **Router de Proveedores**: Implementar un mecanismo que permita ejecutar la misma consulta en **ChatGPT** y **DeepSeek**.
4.  **Orquestación RAG**:
    * **Reescritura de consulta** (opcional).
    * **Recuperación top-k**.
    * **Re-rank** (opcional).
    * **Síntesis con citas**, con un formato como `[Nombre del documento, sección/página]`.
5.  **Política de Abstención**: Si no hay soporte en las fuentes, el asistente debe responder **“No encontrado en normativa UFRO”** y sugerir la oficina o unidad correspondiente.
6.  **Trazabilidad**: Cada respuesta debe incluir **citas** con **enlace o referencia** y **número de página/sección**.
7.  **Métricas**:
    * Crear un **conjunto de 20 preguntas realistas (gold set)**.
    * Calcular la **Exactitud** (EM/semántica), **Cobertura** (tasa de cita), **Precisión@k** del recuperador, **Latencia** y **Costo** (tokens × tarifa estimada).
8.  **Demo**: Proveer una **interfaz de línea de comandos (CLI)** en modo interactivo y un *script* para evaluación en *batch*. Se puede incluir opcionalmente una mini-UI con Gradio o Streamlit.

---

## 🛠️ Plan de Trabajo (10 horas estimadas)

| Hora(s) | Tarea |
| :--- | :--- |
| **0.5** | **Setup**: Creación de repositorio, entorno virtual y archivos de configuración. |
| **0.5** | **Datos**: Descarga de 6-10 documentos de la UFRO y registro de metadatos. |
| **1** | **Ingesta & Chunking**: Extracción de texto de los PDFs, limpieza y fragmentación. |
| **1** | **Embeddings & FAISS**: Creación de los *embeddings* con `sentence-transformers` y construcción del índice FAISS. |
| **1** | **Proveedores LLM**: Implementación de adaptadores para las APIs de ChatGPT y DeepSeek. |
| **1** | **Orquestación RAG**: Desarrollo del *pipeline* RAG, incluyendo la lógica de recuperación y síntesis. |
| **1** | **Evaluación v1**: Creación del *gold set* de preguntas y un *script* de evaluación inicial. |
| **0.5** | **Métricas de Costo/Latencia**: Medición del rendimiento y costo de las consultas. |
| **0.5** | **Comparativa y Análisis**: Comparación de los resultados de ChatGPT y DeepSeek. |
| **1** | **Robustez**: Implementación de la política de abstención y pruebas de alucinación. |
| **1** | **Demo y README**: Creación de la CLI interactiva y la redacción final del README. |
| **1** | **Informe y Limpieza**: Redacción del informe técnico y revisión del código. |

---

## 📂 Estructura del Repositorio
ufro-assistant/
├─ app.py                 # CLI principal (proveedor, k, modo)
├─ providers/
│   ├─ base.py            # Protocolo Provider
│   ├─ chatgpt.py         # Adapter ChatGPT
│   └─ deepseek.py        # Adapter DeepSeek
├─ rag/
│   ├─ ingest.py          # Extracción PDF + chunking + metadatos
│   ├─ embed.py           # Embeddings + FAISS
│   ├─ retrieve.py        # Búsqueda vectorial y re-rank opcional
│   └─ prompts.py         # Plantillas de sistema/usuario
├─ eval/
│   ├─ gold_set.jsonl     # 20 preguntas con respuesta esperada y refs
│   └─ evaluate.py        # Métricas (EM, sim, citas, prec@k)
├─ data/
│   ├─ raw/               # PDFs/HTML
│   ├─ processed/         # chunks.parquet
│   └─ index.faiss
├─ scripts/
│   └─ batch_demo.sh
├─ .env.example
├─ README.md
└─ requirements.txt