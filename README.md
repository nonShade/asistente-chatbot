# UFRO Chatbot - Asistente de Normativa Universitaria 🤖

Sistema de chatbot inteligente que utiliza tecnología RAG (Retrieval-Augmented Generation) para consultar la normativa de la Universidad de La Frontera (UFRO).

## 🚀 Uso

### Instalación
```bash
git clone <repository-url>
cd asistente-chatbot
cp .env.example .env
# Editar .env con tus API keys
./setup.sh
```

### Ejecutar el chatbot
```bash
# Opción 1: Script de conveniencia
./activate.sh
python app.py

# Opción 2: Manual
source .venv/bin/activate
USE_QDRANT=true QDRANT_HOST=localhost python app.py
```

### Comandos especiales en el chat
- `/compare <pregunta>` - Comparar respuestas de ambos proveedores
- `/deepseek <pregunta>` - Usar solo DeepSeek
- `/chatgpt <pregunta>` - Usar solo ChatGPT
- `help` - Mostrar ayuda
- `salir` - Terminar sesión

## ⚙️ Parámetros

### Variables de entorno (.env)
```env
# APIs de Modelos de Lenguaje (al menos una requerida)
DEEPSEEK_API_KEY=tu_deepseek_api_key_aqui
DEEPSEEK_MODEL=deepseek-chat

OPENAI_API_KEY=tu_openai_api_key_aqui  
OPENAI_MODEL=openai/gpt-4o-mini

# Configuración RAG
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=1200
CHUNK_OVERLAP=250

# Qdrant
USE_QDRANT=true
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Modelos de embedding disponibles
- `paraphrase-multilingual-MiniLM-L12-v2` (Recomendado)
- `all-MiniLM-L6-v2` (Rápido)
- `paraphrase-multilingual-mpnet-base-v2` (Mejor calidad)

### Configuración de chunks
- `CHUNK_SIZE`: Tamaño de chunk en caracteres (recomendado: 1200)
- `CHUNK_OVERLAP`: Superposición entre chunks (recomendado: 250)

## ⚠️ Limitaciones

### Técnicas
- **Dependencias**: Requiere Docker para Qdrant y Python 3.11+ con pip
- **Memoria**: Los modelos de embedding requieren ~500MB de RAM
- **Almacenamiento**: Índices vectoriales ocupan ~200MB por cada 100 documentos
- **Red**: Requiere conexión a internet para APIs de LLM

### Funcionales
- **Idioma**: Optimizado para español, funcionalidad limitada en otros idiomas
- **Contexto**: Respuestas basadas únicamente en documentos proporcionados
- **Actualización**: Los índices deben reconstruirse manualmente tras añadir documentos
- **Precisión**: Las respuestas pueden contener imprecisiones, siempre verificar con fuentes oficiales

### Costo
- **DeepSeek**: ~$0.14 por 1M tokens de entrada, ~$0.28 por 1M tokens de salida
- **OpenAI GPT-4o-mini**: ~$0.15 por 1M tokens de entrada, ~$0.60 por 1M tokens de salida
- **Embeddings**: Procesamiento local sin costo adicional

## 📚 Fuentes

### Documentos incluidos
El sistema está entrenado con los siguientes documentos oficiales de la UFRO:

1. **Reglamento de Régimen de Estudios 2023**
   - Normativa académica general
   - Requisitos de matrícula y permanencia

2. **Resolución Exenta 3542-2022 - Reglamento de Admisión para Carreras de Pregrado**
   - Procesos de admisión
   - Requisitos de ingreso

3. **Resolución Exenta 2022326308 - Obligaciones Financieras**
   - Aranceles y costos
   - Políticas de pago

4. **Reglamento de Convivencia**
   - Normas de conducta estudiantil
   - Procedimientos disciplinarios

5. **Manual del Estudiante - Apelaciones 2024**
   - Procesos de apelación académica
   - Procedimientos de reclamo

6. **Preguntas Frecuentes - Beneficios Estudiantiles**
   - Becas y ayudas estudiantiles
   - Requisitos y postulaciones

7. **Reglamento de Actividad de Titulación**
   - Procesos de titulación
   - Modalidades de graduación

8. **Resoluciones Complementarias (3332, 3280, 418)**
   - Normativas específicas adicionales
   - Actualizaciones reglamentarias

### Responsabilidad
- **Información oficial**: Siempre consultar documentos oficiales para decisiones importantes
- **Actualización**: Los documentos corresponden a versiones específicas y pueden haber cambiado
- **Interpretación**: Este sistema es una herramienta de consulta, no un asesor oficial
