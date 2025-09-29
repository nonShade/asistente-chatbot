"""
Script para migrar datos de FAISS a Qdrant.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.qdrant_client import UFROQdrantClient
from rag.embedding_system import EmbeddingSystem


def load_faiss_data(index_path: str, chunks_path: str) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Carga los datos existentes de FAISS.
    
    Args:
        index_path: Ruta al archivo .faiss
        chunks_path: Ruta al archivo .parquet con chunks
        
    Returns:
        Tuple con embeddings y chunks
    """
    print("📚 Cargando datos de FAISS...")
    
    # Cargar chunks desde parquet
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"No se encontró el archivo de chunks: {chunks_path}")
    
    chunks_df = pd.read_parquet(chunks_path)
    print(f"✓ Cargados {len(chunks_df)} chunks desde parquet")
    
    # Convertir DataFrame a lista de diccionarios
    chunks = []
    for _, row in chunks_df.iterrows():
        chunk = {
            "text": row.get("text", ""),
            "title": row.get("title", ""),
            "filename": row.get("filename", ""),
            "page": row.get("page", 0),
            "doc_id": row.get("doc_id", ""),
            "url": row.get("url", ""),
            "vigencia": row.get("vigencia", 2024)
        }
        chunks.append(chunk)
    
    # Regenerar embeddings usando el mismo modelo
    print("🔄 Regenerando embeddings...")
    embedding_system = EmbeddingSystem()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_system.embed_texts(texts)
    
    print(f"✓ Generados embeddings para {len(embeddings)} textos")
    return embeddings, chunks


def migrate_to_qdrant(faiss_index_path: str, chunks_path: str, qdrant_host: str = "localhost"):
    """
    Migra los datos de FAISS a Qdrant.
    
    Args:
        faiss_index_path: Ruta al índice FAISS
        chunks_path: Ruta a los chunks en parquet
        qdrant_host: Host de Qdrant
    """
    print("🚀 Iniciando migración FAISS → Qdrant")
    
    # Verificar que los archivos existen
    if not os.path.exists(chunks_path):
        print(f"❌ No se encontró: {chunks_path}")
        print("💡 Ejecuta primero: python app.py --mode build-index")
        return False
    
    try:
        # 1. Cargar datos de FAISS
        embeddings, chunks = load_faiss_data(faiss_index_path, chunks_path)
        
        # 2. Conectar a Qdrant
        print("🔌 Conectando a Qdrant...")
        qdrant_client = UFROQdrantClient(host=qdrant_host)
        
        # Verificar conexión
        if not qdrant_client.health_check():
            print("❌ No se puede conectar a Qdrant")
            print("💡 Ejecuta: docker-compose up -d")
            return False
        
        # 3. Inicializar colección
        vector_size = embeddings.shape[1] if len(embeddings) > 0 else 384
        qdrant_client.initialize_collection(vector_size=vector_size)
        
        # 4. Insertar datos
        if len(chunks) > 0:
            qdrant_client.insert_documents(chunks, embeddings)
            
            # 5. Verificar inserción
            info = qdrant_client.get_collection_info()
            print(f"✅ Migración completada:")
            print(f"   - Documentos: {info.get('points_count', 0)}")
            print(f"   - Vectores: {info.get('vectors_count', 0)}")
            
            return True
        else:
            print("⚠️ No hay datos para migrar")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la migración: {str(e)}")
        return False


def test_qdrant_search(query: str = "¿Cuáles son los requisitos de matrícula?"):
    """
    Prueba la búsqueda en Qdrant.
    
    Args:
        query: Consulta de prueba
    """
    print(f"🔍 Probando búsqueda: '{query}'")
    
    try:
        # Conectar a Qdrant
        qdrant_client = UFROQdrantClient()
        
        if not qdrant_client.health_check():
            print("❌ Qdrant no está disponible")
            return
        
        # Generar embedding de la consulta
        embedding_system = EmbeddingSystem()
        query_embedding = embedding_system.embed_texts([query])[0]
        
        # Buscar documentos similares
        results = qdrant_client.search_similar(query_embedding, limit=3)
        
        print(f"📋 Encontrados {len(results)} resultados:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (página {result['page']})")
            print(f"     Score: {result['score']:.3f}")
            print(f"     Texto: {result['text'][:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error en la búsqueda: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migración FAISS → Qdrant")
    parser.add_argument("--action", choices=["migrate", "test"], 
                       default="migrate", help="Acción a realizar")
    parser.add_argument("--index-path", default="data/processed/index.faiss",
                       help="Ruta al índice FAISS")
    parser.add_argument("--chunks-path", default="data/processed/chunks.parquet",
                       help="Ruta a los chunks")
    parser.add_argument("--qdrant-host", default="localhost",
                       help="Host de Qdrant")
    
    args = parser.parse_args()
    
    if args.action == "migrate":
        success = migrate_to_qdrant(args.index_path, args.chunks_path, args.qdrant_host)
        if success:
            print("🎉 ¡Migración exitosa! Ahora puedes usar Qdrant")
        else:
            print("💥 La migración falló")
    elif args.action == "test":
        test_qdrant_search()