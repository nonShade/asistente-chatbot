"""
Cliente Qdrant personalizado para el sistema RAG de UFRO.
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
from sentence_transformers import SentenceTransformer


class UFROQdrantClient:
    """Cliente Qdrant especializado para documentos UFRO."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6333,
                 collection_name: str = "ufro_documents"):
        """
        Inicializa el cliente Qdrant.
        
        Args:
            host: Host del servidor Qdrant
            port: Puerto del servidor Qdrant  
            collection_name: Nombre de la colección
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embedding_model = None
        
    def initialize_collection(self, vector_size: int = 384):
        """
        Crea la colección si no existe.
        
        Args:
            vector_size: Dimensión de los vectores (384 para all-MiniLM-L6-v2)
        """
        try:
            # Verificar si la colección existe
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                print(f"🔨 Creando colección '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Colección '{self.collection_name}' creada")
            else:
                print(f"✓ Colección '{self.collection_name}' ya existe")
                
        except Exception as e:
            raise RuntimeError(f"Error inicializando colección: {str(e)}")
    
    def insert_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Inserta documentos y sus embeddings en Qdrant.
        
        Args:
            chunks: Lista de chunks con metadatos
            embeddings: Array numpy con los embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("El número de chunks debe coincidir con el número de embeddings")
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk.get("content", ""),
                    "title": chunk.get("title", ""),
                    "filename": chunk.get("filename", ""),
                    "page": chunk.get("page", 0),
                    "doc_id": chunk.get("doc_id", ""),
                    "url": chunk.get("url", ""),
                    "vigencia": chunk.get("vigencia", 2024),
                    "chunk_index": i
                }
            )
            points.append(point)
        
        # Insertar en lotes para mejor rendimiento
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total_inserted += len(batch)
            print(f"📚 Insertados {total_inserted}/{len(points)} documentos")
        
        print(f"✓ {len(points)} documentos insertados en Qdrant")
    
    def search_similar(self, 
                      query_embedding: np.ndarray, 
                      limit: int = 5,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Busca documentos similares usando el embedding de la consulta.
        
        Args:
            query_embedding: Vector de la consulta
            limit: Número máximo de resultados
            filters: Filtros opcionales (ej: {"vigencia": 2024})
            
        Returns:
            Lista de documentos similares con scores
        """
        # Construir filtros de Qdrant si se proporcionan
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Realizar búsqueda
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=qdrant_filter,
            limit=limit
        )
        
        # Formatear resultados
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "title": hit.payload.get("title", ""),
                "filename": hit.payload.get("filename", ""),
                "page": hit.payload.get("page", 0),
                "doc_id": hit.payload.get("doc_id", ""),
                "url": hit.payload.get("url", ""),
                "vigencia": hit.payload.get("vigencia", 2024)
            }
            results.append(result)
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Obtiene información sobre la colección."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self):
        """Elimina la colección completa."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️ Colección '{self.collection_name}' eliminada")
        except Exception as e:
            print(f"❌ Error eliminando colección: {str(e)}")
    
    def health_check(self) -> bool:
        """Verifica si Qdrant está funcionando."""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            print(f"❌ Qdrant no disponible: {str(e)}")
            return False