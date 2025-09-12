import os
import numpy as np
import faiss
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from scripts.ingest_documents import DocumentChunk


class EmbeddingSystem:
    """Maneja embeddings de documentos y operaciones del índice FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Crea embeddings para los chunks de documentos."""
        texts = [chunk.content for chunk in chunks]
        print(f"Creando embeddings para {len(texts)} chunks...")

        # Genera embeddings en lotes para evitar problemas de memoria
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Construye índice FAISS a partir de los embeddings."""
        dimension = embeddings.shape[1]

        # Usa IndexFlatIP para similitud coseno (después de normalización)
        index = faiss.IndexFlatIP(dimension)

        # Normaliza embeddings para similitud coseno
        faiss.normalize_L2(embeddings)

        # Agrega embeddings al índice
        index.add(embeddings.astype('float32'))

        print(f"Índice FAISS construido con {index.ntotal} vectores")
        return index

    def save_index(self, index: faiss.Index, chunks: List[DocumentChunk],
                   index_path: str, chunks_path: str):
        """Guarda índice FAISS y metadatos de chunks."""
        # Guarda índice FAISS
        faiss.write_index(index, index_path)

        # Guarda metadatos de chunks como parquet
        data = []
        for chunk in chunks:
            data.append({
                'doc_id': chunk.doc_id,
                'title': chunk.title,
                'content': chunk.content,
                'page': chunk.page,
                'chunk_id': chunk.chunk_id,
                'url': chunk.url,
                'vigencia': chunk.vigencia
            })

        df = pd.DataFrame(data)
        df.to_parquet(chunks_path, index=False)

        print(f"Índice guardado en {index_path} y chunks en {chunks_path}")

    def load_index(self, index_path: str, chunks_path: str):
        """Carga índice FAISS y metadatos de chunks."""
        self.index = faiss.read_index(index_path)
        chunks_df = pd.read_parquet(chunks_path)

        self.chunks = []
        for _, row in chunks_df.iterrows():
            chunk = DocumentChunk(
                doc_id=row['doc_id'],
                title=row['title'],
                content=row['content'],
                page=row['page'],
                chunk_id=row['chunk_id'],
                url=row['url'],
                vigencia=row['vigencia']
            )
            self.chunks.append(chunk)

        print(f"Índice cargado con {self.index.ntotal} vectores y {len(self.chunks)} chunks")

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Busca chunks similares usando la consulta."""
        if self.index is None:
            raise ValueError("Índice no cargado. Llama load_index() primero.")

        # Codifica la consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normaliza para similitud coseno
        faiss.normalize_L2(query_embedding)

        # Busca
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        # Retorna resultados con chunks y scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Índice válido
                results.append((self.chunks[idx], float(score)))

        return results

    def build_and_save_index(self, chunks: List[DocumentChunk],
                           index_path: str, chunks_path: str):
        """Pipeline completo: crear embeddings, construir índice y guardar."""
        embeddings = self.create_embeddings(chunks)
        index = self.build_faiss_index(embeddings)
        self.save_index(index, chunks, index_path, chunks_path)

        # Mantiene referencias para uso inmediato
        self.index = index
        self.chunks = chunks
