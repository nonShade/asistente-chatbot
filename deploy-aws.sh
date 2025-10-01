#!/bin/bash

# AWS EC2 Deployment Script for UFRO Chatbot (Production-Ready)
# Usage: ./deploy-aws-prebuilt.sh
# Asume que los índices ya están pre-construidos en el repo

set -e

echo "🚀 Iniciando despliegue en AWS EC2 (con índices pre-construidos)..."

# Verificar que Docker esté funcionando
echo "🔍 Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Instalando Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "⚠️ Docker instalado. Puede que necesites reiniciar la sesión para usar Docker sin sudo."
elif ! docker info &> /dev/null; then
    echo "🔧 Docker está instalado pero no está ejecutándose. Iniciando..."
    sudo systemctl start docker
else
    echo "✅ Docker está funcionando correctamente"
fi

echo "🔍 Verificando Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Instalando Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo "✅ Docker Compose está disponible"
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

# Verificar que los índices pre-construidos existan
echo "🔍 Verificando índices pre-construidos..."
if [ ! -f "data/processed/index.faiss" ] || [ ! -f "data/processed/chunks.parquet" ]; then
    echo "❌ Índices pre-construidos no encontrados!"
    echo "💡 Ejecuta primero en tu máquina local:"
    echo "   python scripts/build_index.py"
    echo "   git add data/processed/"
    echo "   git commit -m 'Add pre-built indices'"
    echo "   git push"
    exit 1
else
    echo "✅ Índices pre-construidos encontrados"
    echo "   - data/processed/index.faiss ($(du -h data/processed/index.faiss | cut -f1))"
    echo "   - data/processed/chunks.parquet ($(du -h data/processed/chunks.parquet | cut -f1))"
fi

# Detener servicios existentes
echo "🛑 Deteniendo servicios existentes..."
docker-compose -f docker-compose.prod.yml down --remove-orphans || true

# Limpiar imágenes antiguas para liberar espacio
echo "🧹 Limpiando imágenes Docker antiguas..."
docker system prune -f

# Construir e iniciar servicios
echo "🔨 Construyendo e iniciando servicios..."
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Verificar el estado de los servicios
echo "⏳ Esperando que los servicios se inicialicen..."
sleep 15

echo "📊 Estado de los servicios:"
docker-compose -f docker-compose.prod.yml ps

# Esperar a que Qdrant esté completamente listo
echo "⏳ Esperando a que Qdrant esté disponible..."
for i in {1..20}; do
    if docker-compose -f docker-compose.prod.yml exec -T qdrant curl -f http://localhost:6333/collections &>/dev/null; then
        echo "✅ Qdrant está listo"
        break
    fi
    echo "⏳ Esperando Qdrant... ($i/20)"
    sleep 3
done

# Migrar índices pre-construidos a Qdrant
echo "🔄 Migrando índices pre-construidos a Qdrant..."
echo "⏳ Este proceso debería tomar menos de 1 minuto..."

if docker-compose -f docker-compose.prod.yml exec -T ufro-chatbot python scripts/migrate_to_qdrant.py --qdrant-host qdrant; then
    echo "✅ Índices migrados exitosamente a Qdrant"
else
    echo "⚠️ Error migrando a Qdrant. La aplicación funcionará con FAISS como fallback."
fi

# Verificación final optimizada
echo "🔍 Verificación final del sistema..."
sleep 3

docker-compose -f docker-compose.prod.yml exec -T ufro-chatbot python -c "
import sys
import time
sys.path.append('/app')
try:
    time.sleep(2)
    from rag.qdrant_client import UFROQdrantClient
    client = UFROQdrantClient()
    if client.health_check():
        collections = client.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f'✅ Qdrant funcionando. Colecciones: {collection_names}')
        if 'ufro_documents' in collection_names:
            info = client.client.get_collection('ufro_documents')
            print(f'✅ Colección ufro_documents: {info.vectors_count} vectores')
        print('✅ Sistema completamente listo para consultas')
    else:
        print('⚠️ Qdrant no responde, usando FAISS como fallback')
except Exception as e:
    print(f'⚠️ Error en verificación: {e}')
    print('⚠️ Usando FAISS como fallback')
" || echo "⚠️ Verificación falló, pero el sistema debería funcionar"

# Mostrar logs de inicio
echo "📋 Últimos logs del chatbot:"
docker-compose -f docker-compose.prod.yml logs --tail=15 ufro-chatbot

echo ""
echo "✅ Despliegue completado!"
echo ""
echo "🌐 El chatbot está disponible en:"
echo "   http://$(curl -s http://checkip.amazonaws.com):5000"
echo ""
echo "📊 Estadísticas del sistema:"
echo "   - Tiempo de deploy: ~1-2 minutos"
echo "   - Índices: Pre-construidos localmente"
echo "   - Base de datos: Qdrant + FAISS fallback"
echo ""
echo "🔍 Para ver logs en tiempo real:"
echo "   docker-compose -f docker-compose.prod.yml logs -f ufro-chatbot"
echo ""
echo "🛑 Para detener los servicios:"
echo "   docker-compose -f docker-compose.prod.yml down"