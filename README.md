# Proyecto Práctico 1: Asistente CHATBOT sobre normativa y reglamentos de la UFRO

## 🚀 Descripción del Proyecto

Este proyecto consiste en el desarrollo de un **asistente conversacional RAG (Retrieval-Augmented Generation)** en Python. El objetivo principal es responder preguntas de estudiantes y personal sobre la normativa y reglamentos de la Universidad de La Frontera (UFRO), citando las fuentes oficiales.

El asistente integra múltiples proveedores de modelos de lenguaje grandes (LLM): **Gemini**, **DeepSeek** y **ChatGPT**, mediante un **patrón de proveedor** que permite la comparación (A/B) o el consenso (ensemble) de sus respuestas.

### Tecnologías Utilizadas
* **Python 3.11+**
* **Google Gemini API**
* **DeepSeek API**
* **OpenAI ChatGPT API** (implementada a pesar de no posser "API_KEY", por ende no se conoce la resolucion del proveedor)
* **RAG** con **FAISS**
* **Sentence Transformers**
* **PyPDF** para extracción de texto
* **Rich** para interfaz CLI mejorada

## 📦 Instalación

### Prerrequisitos
- Python 3.11 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio:**
```bash
git clone <url-del-repositorio>
cd asistente-chatbot
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
```bash
cp .env.example .env
# Editar .env con tus claves API
```

### Configuración de APIs

Edita el archivo `.env` con tus claves API:

```env
GEMINI_API_KEY=tu_clave_gemini_aqui
DEEPSEEK_API_KEY=tu_clave_deepseek_aqui
OPENAI_API_KEY=tu_clave_openai_aqui  # Opcional

# Configuración de modelos (opcional)
GEMINI_MODEL=gemini-1.5-flash
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Configuración de chunking
CHUNK_SIZE=900
CHUNK_OVERLAP=120
```

## 🚀 Uso

### Modo Interactivo (CLI)

```bash
python app.py
```

### Comandos Especiales de la CLI

- `/compare <pregunta>` - Comparar respuestas de todos los proveedores
- `/gemini <pregunta>` - Usar solo Gemini
- `/deepseek <pregunta>` - Usar solo DeepSeek
- `/eval` - Ejecutar evaluación completa
- `help` - Mostrar ayuda
- `salir` - Terminar sesión

### Ejemplos de Uso

```bash
# Pregunta general
¿Cuáles son los requisitos para la matrícula?

# Comparar proveedores
/compare ¿Qué dice el reglamento sobre las apelaciones?

# Usar proveedor específico
/gemini ¿Cuándo son las fechas de matrícula 2025?
```

### Modo de Evaluación

```bash
python app.py --eval
```

Ejecuta la evaluación completa usando el conjunto de preguntas en `eval/gold_questions.csv`.

## ⚙️ Parámetros y Configuración

### Variables de Entorno

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `GEMINI_API_KEY` | Clave API de Google Gemini | **Requerida** |
| `DEEPSEEK_API_KEY` | Clave API de DeepSeek | **Requerida** |
| `OPENAI_API_KEY` | Clave API de OpenAI | Opcional |
| `GEMINI_MODEL` | Modelo de Gemini a usar | `gemini-1.5-flash` |
| `DEEPSEEK_MODEL` | Modelo de DeepSeek a usar | `deepseek-chat` |
| `EMBEDDING_MODEL` | Modelo para embeddings | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Tamaño de chunks de texto | `900` |
| `CHUNK_OVERLAP` | Solapamiento entre chunks | `120` |

## ⚠️ Limitaciones

### Limitaciones Técnicas
- **Dependencia de APIs externas**: Requiere conectividad a internet y claves API válidas
- **Idioma**: Optimizado para consultas en español sobre normativa UFRO
- **Contexto limitado**: Máximo ~4000 tokens por consulta (limitación de modelos)
- **Precisión temporal**: La información está limitada a los documentos ingresados (vigencia hasta 2025)

### Limitaciones de Contenido
- **Cobertura parcial**: Solo incluye documentos oficiales procesados
- **Interpretación legal**: No constituye asesoría legal oficial
- **Actualización manual**: Requiere reingesta de documentos para actualizaciones
- **Consultas complejas**: Puede tener dificultades con preguntas que requieren razonamiento complejo

### Limitaciones de Rendimiento
- **Latencia**: Tiempo de respuesta variable según la API utilizada (2-10 segundos)
- **Costo**: Uso de tokens limitado por presupuesto de APIs
- **Concurrencia**: Sin soporte para múltiples usuarios simultáneos
- **Memoria**: Índice FAISS cargado en memoria (~100MB)

## 📊 Fuentes de Datos

El sistema utiliza los siguientes documentos oficiales de la UFRO:

### Documentos de Normativa (2022-2025)

| Documento | Descripción | Vigencia | Fuente |
|-----------|-------------|----------|---------|
| **Reglamento de Régimen de Estudios** | Normativa académica general | 2023 | [ufro.cl/normativa](https://www.ufro.cl/normativa/) |
| **Reglamento de Admisión** | Proceso de admisión pregrado | 2022 | [ufro.cl/normativa](https://www.ufro.cl/normativa/) |
| **Obligaciones Financieras** | Aranceles y pagos | 2022 | [ufro.cl/normativa](https://www.ufro.cl/normativa/) |
| **Reglamento de Convivencia** | Normas de convivencia universitaria | 2023 | [ufro.cl/normativa](https://www.ufro.cl/normativa/) |
| **Reglamento de Titulación** | Procesos de graduación | 2023 | [ufro.cl/normativa](https://www.ufro.cl/normativa/) |

### Documentos de Procedimientos (2024-2025)

| Documento | Descripción | Vigencia | Fuente |
|-----------|-------------|----------|---------|
| **Información de Matrícula** | Guía proceso matrícula | 2024 | [ufro.cl/estudiantes](https://www.ufro.cl/estudiantes/) |
| **Manual de Apelaciones** | Procedimientos de apelación | 2024 | [ufro.cl/estudiantes](https://www.ufro.cl/estudiantes/) |
| **Beneficios Estudiantiles** | Preguntas frecuentes beneficios | 2024 | [ufro.cl/estudiantes](https://www.ufro.cl/estudiantes/) |
| **Calendario Académico** | Fechas importantes 2025 | 2025 | [ufro.cl/calendario](https://www.ufro.cl/calendario-academico/) |

### Estadísticas del Corpus
- **Total documentos**: 11 archivos oficiales
- **Formato**: PDF y texto plano
- **Chunks procesados**: ~200-300 fragmentos
- **Periodo cubierto**: 2022-2025
- **Idioma**: Español
- **Tamaño total**: ~15MB de texto

### Metadatos Incluidos
Para cada documento se mantiene:
- **doc_id**: Identificador único
- **título**: Nombre oficial
- **filename**: Archivo original
- **url**: Fuente de descarga
- **vigencia**: Año de validez

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
