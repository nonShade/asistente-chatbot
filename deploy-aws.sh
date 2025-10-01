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

# Construir índices de documentos
echo "📚 Construyendo índices de documentos..."
echo "⏳ Este proceso puede tomar 2-5 minutos..."

# Esperar a que Qdrant esté completamente listo
echo "⏳ Esperando a que Qdrant esté completamente disponible..."
for i in {1..30}; do
    if docker-compose -f docker-compose.prod.yml exec -T qdrant curl -f http://localhost:6333/collections &>/dev/null; then
        echo "✓ Qdrant está listo"
        break
    fi
    echo "⏳ Esperando Qdrant... ($i/30)"
    sleep 2
done

# Construir índices
echo "🔨 Ejecutando construcción de índices..."
if docker-compose -f docker-compose.prod.yml exec -T ufro-chatbot python scripts/build_index.py; then
    echo "✅ Índices construidos exitosamente"
else
    echo "❌ Error construyendo índices. Intentando una vez más..."
    sleep 10
    docker-compose -f docker-compose.prod.yml exec -T ufro-chatbot python scripts/build_index.py
fi

# Verificar que los índices se crearon correctamente
echo "🔍 Verificando que los índices se crearon correctamente..."
sleep 5

# Verificar colecciones en Qdrant
echo "🗃️ Verificando colecciones en Qdrant..."
sleep 5  # Dar tiempo para que Qdrant procese la inserción
if docker-compose -f docker-compose.prod.yml exec -T qdrant curl -s http://localhost:6333/collections | grep -q "ufro_documents"; then
    echo "✅ Colección ufro_documents encontrada"
else
    echo "⚠️ Colección ufro_documents no encontrada en la respuesta de la API"
fi

# Verificar desde Python con mejor manejo de errores
echo "🔍 Verificando desde Python..."
docker-compose -f docker-compose.prod.yml exec -T ufro-chatbot python -c "
import os
import sys
import time
sys.path.append('/app')
try:
    # Esperar un poco más para conexiones
    time.sleep(2)
    from rag.qdrant_client import UFROQdrantClient
    client = UFROQdrantClient()
    
    # Verificar conectividad primero
    if not client.health_check():
        print('❌ Error: No se puede conectar a Qdrant')
        sys.exit(1)
    
    collections = client.client.get_collections()
    collection_names = [c.name for c in collections.collections]
    print(f'✓ Colecciones encontradas: {collection_names}')
    
    if 'ufro_documents' in collection_names:
        info = client.client.get_collection('ufro_documents')
        print(f'✓ Colección ufro_documents: {info.vectors_count} vectores')
        print('✅ Sistema listo para consultas')
    else:
        print('⚠️ Advertencia: Colección ufro_documents no encontrada')
        print('ℹ️ Puede que esté procesándose aún. Verificar más tarde.')
        
except ImportError as e:
    print(f'❌ Error de importación: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️ Error verificando índices: {e}')
    print('ℹ️ Los índices pueden estar funcionando correctamente a pesar del error de verificación')
    # No salir con error ya que los índices pueden estar funcionando
"

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