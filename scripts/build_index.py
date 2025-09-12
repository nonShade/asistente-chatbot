#!/usr/bin/env python3
"""
Script para procesar documentos y construir el índice FAISS.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest_documents import DocumentIngester
from rag.embedding_system import EmbeddingSystem

def main():
    """Procesar documentos y construir índice"""
    print("Iniciando procesamiento de documentos...")

    # Configuración para extracción
    chunk_size = 800 # Chunks más pequeños para mejor contexto
    chunk_overlap = 120 # Mayor overlap para mejorar la continuidad
    data_dir = 'data/raw'
    sources_file = 'data/sources.csv'

    # Verificar que existan los datos
    if not os.path.exists(data_dir):
        print(f"Directorio de datos no encontrado: {data_dir}")
        return

    if not os.path.exists(sources_file):
        print(f"Archivo de fuentes no encontrado: {sources_file}")
        return

    # Paso 1: Ingesta y procesamiento de documentos
    print(f"Procesando documentos en {data_dir}...")
    ingester = DocumentIngester(chunk_size, chunk_overlap)
    chunks = ingester.process_documents(data_dir, sources_file)

    if not chunks:
        print("No se encontraron chunks para procesar.")
        return

    print(f"Procesados {len(chunks)} chunks")

    # Paso 2: Embedding y FAISS
    print("Generando embeddings y construyendo índice FAISS...")
    embedding_system = EmbeddingSystem()

    # Crear directorios si no existen
    os.makedirs("data/processed", exist_ok=True)

    # Construir y guardar el índice FAISS
    embedding_system.build_and_save_index(
        chunks,
        "data/processed/index.faiss",
        "data/processed/chunks.parquet"
    )

    print("Índice FAISS construido exitosamente.")
    print(f"Archivos generados:")
    print(f" - data/processed/index.faiss")
    print(f" - data/processed/chunks.parquet")
    print("\n ¡Listo! Ahora puedes ejecutar el asistente con:")
    print("python app.py --mode interactive")

if __name__ == "__main__":
    main()