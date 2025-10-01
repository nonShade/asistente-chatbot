# UFRO Chatbot - Asistente de Normativa Universitaria 🤖

Sistema de chatbot inteligente que utiliza tecnología RAG (Retrieval-Augmented Generation) para consultar la normativa de la Universidad de La Frontera (UFRO).

## 🚀 Uso (desarrollo)

### Instalación
```bash
git clone <repository-url>
cd asistente-chatbot
cp .env.example .env
# Editar .env con tus API keys
pip install -r requirements.txt
```

### Ejecutar el chatbot
```bash
# Iniciar Qdrant con Docker
docker-compose up -d

# Ejecutar el chatbot
python app.py
 
# O con variables específicas
USE_QDRANT=true QDRANT_HOST=localhost python app.py
```

## 🚀 Uso (produccion o instancia aws)
### Instalación
```bash
git clone <repository-url>
cd asistente-chatbot
cp .env.example .env
# Editar .env con tus API keys
```

### Ejecutar script de despliegue
```bash
chmod +x deploy-aws.sh

# Ejecutar script
./deploy-aws.sh
```
### El script automatiza:

1. Instalación de dependencias: Docker y Docker Compose
2. Configuración de entorno: Verificación y creación del archivo .env
3. Construcción de servicios: Chatbot y base de datos vectorial Qdrant
4. Indexación de documentos: Procesamiento automático de PDFs en /data/raw/
5. Verificación de salud: Comprobación de servicios y índices

### Acceso a la aplicación

Una vez completado el despliegue, entrar a la aplicacion web mediante el link de flask que dara el docker-compose luego de que se ejecute completamente el script

### Comandos especiales en el chat (desarrollo)
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

# Configuración RAG (embedding_model limitado debido a los recursos de la instancia)
EMBEDDING_MODEL=all-MiniLM-L6-v2
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

## 🛡️ Ética y Responsabilidad

### Política de Abstención
Este sistema de chatbot tiene las siguientes limitaciones éticas explícitas:

- **No proporciona asesoría legal vinculante**: Las respuestas son informativas y no constituyen asesoría oficial
- **No reemplaza consultas oficiales**: Para decisiones académicas importantes, consultar directamente con las oficinas correspondientes de la UFRO
- **No interpreta casos específicos**: El sistema no debe usarse para resolver situaciones académicas particulares que requieran evaluación humana
- **Se abstiene de**: Dar consejos sobre procedimientos disciplinarios activos, interpretaciones legales específicas, o decisiones que afecten el estatus académico del estudiante

### Vigencia Normativa
- **Documentos base**: Todos los documentos incluidos corresponden a versiones oficiales vigentes al momento de la indexación
- **Responsabilidad de actualización**: Es responsabilidad del usuario verificar la vigencia actual de la normativa antes de tomar decisiones
- **Advertencia temporal**: La información puede haber sido actualizada después de la última indexación del sistema

### Privacidad
- **No almacenamiento de consultas**: Las conversaciones no se almacenan permanentemente en el sistema
- **Datos temporales**: Solo se mantienen datos en memoria durante la sesión activa
- **APIs externas**: Las consultas se procesan a través de APIs de terceros (OpenAI/DeepSeek) sujetas a sus políticas de privacidad
- **Recomendación**: Evitar compartir información personal identificable en las consultas

## 📋 Tabla de Trazabilidad de Documentos

| doc_id | Título | Archivo | URL Oficial | Vigencia | Estado |
|--------|--------|---------|-------------|----------|---------|
| `reglamento_estudios` | Reglamento de Régimen de Estudios 2023 | 01-Reglamento-de-Regimen-de-Estudios-2023.pdf | [UFRO Normativa](https://www.ufro.cl/normativa/) | 2023 | Vigente |
| `reglamento_admision` | Reglamento de Admisión para Carreras de Pregrado | 02-Res-Ex-3542-2022-Reglamento-de-Admision-para-carreras-de-Pregrado.pdf | [UFRO Normativa](https://www.ufro.cl/normativa/) | 2022 | Vigente |
| `obligaciones_financieras` | Obligaciones Financieras | 03-resex-2022326308-obligaciones-financieras.pdf | [UFRO Normativa](https://www.ufro.cl/normativa/) | 2022 | Vigente |
| `reglamento_convivencia` | Reglamento de Convivencia | 04-Reglamento-Convivencia-rex.pdf | [UFRO Normativa](https://www.ufro.cl/normativa/) | 2023 | Vigente |
| `info_matricula` | Información de Matrícula | INFO-matricula.pdf | [UFRO Estudiantes](https://www.ufro.cl/estudiantes/) | 2024 | Vigente |
| `manual_apelacion` | Manual del Estudiante Apelación 2024 | manual_del_estudiante_apelacion_2024.pdf | [UFRO Estudiantes](https://www.ufro.cl/estudiantes/) | 2024 | Vigente |
| `beneficios_estudiantiles` | Preguntas Frecuentes Beneficios Estudiantiles | Preguntas-frecuentes-beneficios-estudiantiles.pdf | [UFRO Estudiantes](https://www.ufro.cl/estudiantes/) | 2024 | Vigente |
| `reglamento_titulacion` | Reglamento de Actividad de Titulación | Reglamento_actividad_titulacion.pdf | [UFRO Normativa](https://www.ufro.cl/normativa/) | 2023 | Vigente |
| `res_3332` | Resolución Ex. 3332 | Res. Ex. 3332.pdf | [UFRO Normativa](https://www.ufro.cl/normativa/) | 2023 | Vigente |

### Notas sobre Trazabilidad
- **doc_id**: Identificador único utilizado internamente por el sistema RAG
- **URL Oficial**: Enlaces a las secciones oficiales donde se publican las versiones actualizadas
- **Vigencia**: Año de la versión incluida en el sistema
- **Verificación**: Se recomienda verificar la vigencia en el sitio oficial antes de tomar decisiones importantes

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

### Responsabilidad Extendida
- **Verificación obligatoria**: Para decisiones académicas, administrativas o financieras importantes, verificar siempre con fuentes oficiales actualizadas
- **Limitaciones del sistema**: Este chatbot es una herramienta de consulta inicial, no un sistema de gestión académica oficial
- **Escalamiento necesario**: Casos complejos o específicos deben escalarse a las oficinas competentes de la UFRO
- **Actualización de documentos**: La universidad puede actualizar la normativa sin previo aviso; el sistema refleja el estado al momento de la última indexación
- **Información oficial**: Siempre consultar documentos oficiales para decisiones importantes
- **Interpretación**: Este sistema es una herramienta de consulta, no un asesor oficial
