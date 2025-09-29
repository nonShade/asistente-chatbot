#!/bin/bash

# AWS EC2 Deployment Script for UFRO Chatbot
# Usage: ./deploy-aws.sh

set -e

echo "🚀 Iniciando despliegue en AWS EC2..."

# Verificar que docker y docker-compose estén instalados
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Instalando Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Instalando Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Verificar archivo .env
if [ ! -f .env ]; then
    echo "⚠️ Archivo .env no encontrado. Creando desde .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✏️ Por favor, edita el archivo .env con tus claves API:"
        echo "   - DEEPSEEK_API_KEY"
        echo "   - OPENAI_API_KEY"
        read -p "Presiona Enter cuando hayas configurado las claves..."
    else
        echo "❌ No se encuentra .env.example. Creando .env básico..."
        cat > .env << EOF
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_MODEL=deepseek-chat
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=900
CHUNK_OVERLAP=120
USE_QDRANT=true
QDRANT_HOST=qdrant
QDRANT_PORT=6333
EOF
        echo "✏️ Archivo .env creado. Por favor, edita las claves API y ejecuta el script nuevamente."
        exit 1
    fi
fi

# Detener servicios existentes
echo "🛑 Deteniendo servicios existentes..."
docker-compose -f docker-compose.prod.yml down --remove-orphans || true

# Construir e iniciar servicios
echo "🔨 Construyendo e iniciando servicios..."
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Verificar el estado de los servicios
echo "⏳ Esperando que los servicios se inicialicen..."
sleep 30

echo "📊 Estado de los servicios:"
docker-compose -f docker-compose.prod.yml ps

# Verificar logs
echo "📋 Últimos logs del chatbot:"
docker-compose -f docker-compose.prod.yml logs --tail=20 ufro-chatbot

echo "✅ Despliegue completado!"
echo ""
echo "🌐 El chatbot debería estar disponible en:"
echo "   http://$(curl -s http://checkip.amazonaws.com):5000"
echo ""
echo "🔍 Para ver logs en tiempo real:"
echo "   docker-compose -f docker-compose.prod.yml logs -f ufro-chatbot"
echo ""
echo "🛑 Para detener los servicios:"
echo "   docker-compose -f docker-compose.prod.yml down"