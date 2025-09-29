#!/usr/bin/env python3
"""
Script de prueba rápida para verificar las mejoras en el sistema RAG.
"""

import os
import sys
from pathlib import Path

# Agrega el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from rag.embedding_system import EmbeddingSystem
from rag.rag_system import RAGSystem
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from eval.evaluator import RAGEvaluator

def quick_test():
    """Prueba rápida con una pregunta."""
    print("🧪 Prueba rápida del sistema RAG mejorado")
    
    # Pregunta de prueba
    test_question = "¿Cuáles son los requisitos para matricularse en la UFRO?"
    
    # Configurar un proveedor (usar el que esté disponible)
    providers = []
    
    if deepseek_key := os.getenv("DEEPSEEK_API_KEY"):
        providers.append(DeepSeekProvider(deepseek_key, "deepseek-chat"))
        print("✅ Usando DeepSeek")
    elif openai_key := os.getenv("OPENAI_API_KEY"):
        providers.append(ChatGPTProvider(openai_key, "gpt-4"))
        print("✅ Usando ChatGPT")
    else:
        print("❌ No se encontraron claves API")
        return
    
    # Cargar sistema de embeddings
    print("📚 Cargando sistema de embeddings...")
    embedding_system = EmbeddingSystem()
    
    if not os.path.exists("data/index_improved.faiss"):
        print("❌ Índice FAISS no encontrado")
        return
    
    embedding_system.load_index("data/index_improved.faiss", "data/chunks_improved.parquet")
    
    # Crear sistema RAG
    rag_system = RAGSystem(
        embedding_system=embedding_system,
        providers=providers
    )
    
    # Procesar pregunta
    print(f"\n❓ Pregunta: {test_question}")
    print("⏳ Procesando...")
    
    responses = rag_system.process_query(test_question)
    
    if responses:
        response = responses[0]
        print(f"\n✅ Respuesta de {response.provider_name}:")
        print(f"📝 {response.answer}")
        print(f"\n📊 Métricas:")
        print(f"  - Tokens: {response.tokens_used}")
        print(f"  - Latencia: {response.latency:.2f}s")
        print(f"  - Costo: ${response.cost:.4f}")
        print(f"  - Fuentes: {len(response.sources)}")
        
        # Evaluar calidad
        evaluator = RAGEvaluator(rag_system)
        metrics = evaluator.metrics.evaluate_response_quality(response)
        print(f"  - Score de calidad: {metrics:.3f}")
        
    else:
        print("❌ No se pudo generar respuesta")

if __name__ == "__main__":
    quick_test()