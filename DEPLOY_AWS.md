# Guía de Despliegue AWS EC2 - UFRO Chatbot

## Prerrequisitos

1. **Instancia EC2** con Ubuntu 20.04 o superior
2. **Puertos abiertos** en el Security Group:
   - 22 (SSH)
   - 80 (HTTP)
   - 5000 (Flask App)
   - 6333 (Qdrant)
3. **Claves API** configuradas:
   - DeepSeek API Key
   - OpenAI API Key (opcional)

## Pasos de Instalación

### 1. Conectar a la instancia EC2

```bash
ssh -i tu-clave.pem ubuntu@tu-ip-publica
```

### 2. Actualizar el sistema

```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Instalar Docker y Docker Compose

```bash
# Instalar Docker
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Instalar Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Cerrar sesión y volver a conectar para aplicar cambios de grupo
exit
ssh -i tu-clave.pem ubuntu@tu-ip-publica
```

### 4. Clonar el repositorio

```bash
git clone <tu-repositorio> ufro-chatbot
cd ufro-chatbot
```

### 5. Configurar variables de entorno

```bash
cp .env.example .env
nano .env
```

Configura las siguientes variables:

```bash
DEEPSEEK_API_KEY=tu_clave_deepseek
OPENAI_API_KEY=tu_clave_openai_opcional
DEEPSEEK_MODEL=deepseek-chat
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=900
CHUNK_OVERLAP=120
USE_QDRANT=true
QDRANT_HOST=qdrant
QDRANT_PORT=6333
FLASK_DEBUG=false
SECRET_KEY=tu_clave_secreta_para_produccion
```

### 6. Desplegar con Docker Compose

```bash
# Dar permisos al script de despliegue
chmod +x deploy-aws.sh

# Ejecutar el despliegue
./deploy-aws.sh
```

### 7. Verificar el despliegue

```bash
# Verificar que los contenedores estén corriendo
docker-compose -f docker-compose.prod.yml ps

# Ver logs
docker-compose -f docker-compose.prod.yml logs -f ufro-chatbot

# Verificar salud de la aplicación
curl http://localhost:5000/health
```

## Acceso a la Aplicación

- **Web Interface**: http://tu-ip-publica:5000
- **API Health**: http://tu-ip-publica:5000/health
- **API Chat**: POST http://tu-ip-publica:5000/api/chat

## Comandos Útiles

### Gestión de contenedores

```bash
# Iniciar servicios
docker-compose -f docker-compose.prod.yml up -d

# Detener servicios
docker-compose -f docker-compose.prod.yml down

# Reconstruir e iniciar
docker-compose -f docker-compose.prod.yml up -d --build

# Ver logs en tiempo real
docker-compose -f docker-compose.prod.yml logs -f

# Reiniciar un servicio específico
docker-compose -f docker-compose.prod.yml restart ufro-chatbot
```

### Monitoreo

```bash
# Ver estado de los contenedores
docker ps

# Ver uso de recursos
docker stats

# Ver logs específicos
docker logs ufro-chatbot-prod

# Acceder al contenedor
docker exec -it ufro-chatbot-prod /bin/bash
```

### Mantenimiento

```bash
# Limpiar imágenes no utilizadas
docker image prune -f

# Limpiar volúmenes no utilizados
docker volume prune -f

# Backup de datos de Qdrant
docker cp ufro-qdrant-prod:/qdrant/storage ./qdrant-backup-$(date +%Y%m%d)
```

## Solución de Problemas

### 1. Error de conexión a Qdrant

```bash
# Verificar que Qdrant esté corriendo
docker logs ufro-qdrant-prod

# Reiniciar Qdrant
docker-compose -f docker-compose.prod.yml restart qdrant
```

### 2. Error de claves API

```bash
# Verificar variables de entorno
docker exec ufro-chatbot-prod env | grep API_KEY

# Actualizar variables y reiniciar
nano .env
docker-compose -f docker-compose.prod.yml restart ufro-chatbot
```

### 3. Problemas de memoria

```bash
# Ver uso de memoria
free -h
docker stats

# Reiniciar contenedores si es necesario
docker-compose -f docker-compose.prod.yml restart
```

### 4. Reconstruir índices

```bash
# Acceder al contenedor
docker exec -it ufro-chatbot-prod /bin/bash

# Dentro del contenedor, reconstruir índices
python scripts/build_index.py
```

## Configuración de Dominio (Opcional)

Para usar un dominio en lugar de IP:

1. **Configurar DNS** apuntando a tu IP de EC2
2. **Actualizar nginx.conf** con tu dominio
3. **Configurar SSL** con Let's Encrypt:

```bash
# Instalar certbot
sudo apt install certbot python3-certbot-nginx

# Obtener certificado SSL
sudo certbot --nginx -d tu-dominio.com

# Reiniciar nginx
docker-compose -f docker-compose.prod.yml restart nginx
```

## Backup y Restauración

### Crear backup

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/$DATE

# Backup de datos
cp -r data/ backups/$DATE/
docker cp ufro-qdrant-prod:/qdrant/storage backups/$DATE/qdrant/

# Comprimir
tar -czf backups/ufro-chatbot-backup-$DATE.tar.gz backups/$DATE/
rm -rf backups/$DATE/

echo "Backup creado: ufro-chatbot-backup-$DATE.tar.gz"
```

### Restaurar backup

```bash
#!/bin/bash
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Uso: ./restore.sh backup-file.tar.gz"
    exit 1
fi

# Extraer backup
tar -xzf $BACKUP_FILE

# Detener servicios
docker-compose -f docker-compose.prod.yml down

# Restaurar datos
cp -r backups/*/data/ ./
docker cp backups/*/qdrant/ ufro-qdrant-prod:/qdrant/storage

# Reiniciar servicios
docker-compose -f docker-compose.prod.yml up -d
```

## Actualización de la Aplicación

```bash
# Detener servicios
docker-compose -f docker-compose.prod.yml down

# Actualizar código
git pull origin main

# Reconstruir e iniciar
docker-compose -f docker-compose.prod.yml up -d --build

# Verificar
docker-compose -f docker-compose.prod.yml logs -f ufro-chatbot
```