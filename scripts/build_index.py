#!/usr/bin/env python3
"""
Script para procesar documentos y construir índices FAISS y Qdrant.
"""

import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest_documents import DocumentIngester  
from rag.embedding_system import EmbeddingSystem

# Importar Qdrant solo si está disponible
try:
    from rag.qdrant_client import UFROQdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

def main():
    """Procesar documentos y construir índices"""
    print("🚀 Iniciando construcción de índices...")
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Configuración 
    chunk_size = int(os.getenv('CHUNK_SIZE', '1200'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '250'))
    data_dir = 'data/raw'
    sources_file = 'data/sources.csv'
    use_qdrant = os.getenv('USE_QDRANT', 'true').lower() == 'true'

    # Verificar que existan los datos
    if not os.path.exists(data_dir):
        print(f"❌ Directorio de datos no encontrado: {data_dir}")
        return

    if not os.path.exists(sources_file):
        print(f"❌ Archivo de fuentes no encontrado: {sources_file}")
        return

    # Paso 1: Ingesta y procesamiento de documentos
    print(f"📄 Procesando documentos en {data_dir}...")
    ingester = DocumentIngester(chunk_size, chunk_overlap)
    chunks = ingester.process_documents(data_dir, sources_file)

    if not chunks:
        print("❌ No se encontraron chunks para procesar.")
        return

    print(f"✅ Procesados {len(chunks)} chunks")

    # Paso 2: Crear sistema de embeddings
    print("🔍 Inicializando sistema de embeddings...")
    embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embedding_system = EmbeddingSystem(embedding_model)

    # Crear directorios necesarios
    os.makedirs("data/processed", exist_ok=True)

    # Paso 3: Construir índice FAISS (siempre como respaldo)
    print("📚 Construyendo índice FAISS...")
    embedding_system.build_and_save_index(
        chunks,
        "data/processed/index.faiss", 
        "data/processed/chunks.parquet"
    )
    print("✅ Índice FAISS creado exitosamente")

    # Paso 4: Migrar a Qdrant si está habilitado
    if use_qdrant and QDRANT_AVAILABLE:
        print("🔄 Migrando datos a Qdrant...")
        try:
            # Esperar a que Qdrant esté listo
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
            
            print(f"Conectando a Qdrant en {qdrant_host}:{qdrant_port}...")
            
            # Reintentar conexión varias veces
            max_retries = 30
            qdrant_client = None
            
            for i in range(max_retries):
                try:
                    from rag.qdrant_client import UFROQdrantClient
                    qdrant_client = UFROQdrantClient(host=qdrant_host, port=qdrant_port)
                    if qdrant_client.health_check():
                        print("✅ Conectado a Qdrant")
                        break
                except Exception as e:
                    qdrant_client = None
                    if i < max_retries - 1:
                        print(f"⏳ Esperando a Qdrant... ({i+1}/{max_retries})")
                        time.sleep(2)
                    else:
                        print(f"❌ No se pudo conectar a Qdrant después de {max_retries} intentos")
                        print("ℹ️ Usando solo índice FAISS")
                        return
            
            # Migrar datos a Qdrant si la conexión fue exitosa
            if qdrant_client is not None:
                embeddings = embedding_system.create_embeddings(chunks)
                
                # Determinar el tamaño del vector basado en el modelo de embedding
                vector_size = embeddings.shape[1]
                qdrant_client.initialize_collection(vector_size=vector_size)
                
                # Convertir chunks a formato de diccionario para Qdrant
                chunk_dicts = []
                for chunk in chunks:
                    chunk_dict = {
                        'content': chunk.content,
                        'title': chunk.title,
                        'filename': chunk.title,  # Usar title como filename
                        'page': chunk.page,
                        'doc_id': chunk.doc_id,
                        'url': chunk.url,
                        'vigencia': chunk.vigencia
                    }
                    chunk_dicts.append(chunk_dict)
                
                qdrant_client.insert_documents(chunk_dicts, embeddings)
            print("✅ Datos migrados a Qdrant exitosamente")
            
        except Exception as e:
            print(f"❌ Error migrando a Qdrant: {e}")
            print("ℹ️ Usando solo índice FAISS")

    print("\n🎉 Construcción de índices completada!")
    print("📊 Archivos generados:")
    print("  - data/processed/index.faiss")
    print("  - data/processed/chunks.parquet")
    if use_qdrant and QDRANT_AVAILABLE:
         print("  - Colección en Qdrant: ufro_documents")
    print("\n🚀 Sistema listo para usar!")

if __name__ == "__main__":
    main()