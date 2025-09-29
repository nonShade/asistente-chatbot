#!/bin/bash

echo "🚀 Iniciando UFRO Chatbot (Modo Híbrido)..."

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker no está instalado"
    exit 1
fi

# Verificar archivo .env
if [ ! -f ".env" ]; then
    echo "⚠️  Archivo .env no encontrado. Copiando desde .env.example..."
    cp .env.example .env
    echo "📝 Por favor edita .env con tus API keys."
    read -p "Presiona Enter cuando hayas configurado .env..."
fi

# Crear directorios
echo "📁 Creando directorios necesarios..."
mkdir -p data/qdrant_storage data/processed

# Solo iniciar Qdrant
echo "🗃️  Iniciando Qdrant..."
docker run -d \
  --name ufro-qdrant \
  --network host \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  -e QDRANT__SERVICE__HTTP_PORT=6333 \
  -e QDRANT__SERVICE__GRPC_PORT=6334 \
  --restart unless-stopped \
  qdrant/qdrant:latest

echo "⏳ Esperando a que Qdrant esté listo..."
sleep 10

# Verificar Qdrant
if curl -f http://localhost:6333/ &> /dev/null; then
    echo "✅ Qdrant funcionando correctamente"
else
    echo "❌ Error: Qdrant no responde"
    exit 1
fi

# Configurar entorno virtual Python
echo "🐍 Configurando entorno virtual Python..."
if [ ! -d ".venv" ]; then
    echo "📦 Creando entorno virtual..."
    python -m venv .venv
fi

echo "🔧 Activando entorno virtual e instalando dependencias..."
source .venv/bin/activate
pip install -r requirements.txt

# Construir índices
echo "📚 Construyendo índices de documentos..."
source .venv/bin/activate
USE_QDRANT=true QDRANT_HOST=localhost QDRANT_PORT=6333 python scripts/build_index.py

echo ""
echo "🎉 ¡Sistema listo!"
echo ""
echo "Para usar el chatbot:"
echo "  source .venv/bin/activate"
echo "  USE_QDRANT=true QDRANT_HOST=localhost python app.py"
echo ""
echo "Para detener Qdrant:"
echo "  docker stop ufro-qdrant && docker rm ufro-qdrant"