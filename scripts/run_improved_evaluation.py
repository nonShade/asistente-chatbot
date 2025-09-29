#!/usr/bin/env python3
"""
Script para ejecutar evaluación completa del sistema RAG mejorado.
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

def setup_providers():
    """Configura los proveedores LLM."""
    providers = []
    
    # ChatGPT/OpenRouter
    if openai_key := os.getenv("OPENAI_API_KEY"):
        providers.append(ChatGPTProvider(openai_key, "gpt-4"))
        print("✅ ChatGPT configurado")
    else:
        print("⚠️ OPENAI_API_KEY no encontrado")
    
    # DeepSeek
    if deepseek_key := os.getenv("DEEPSEEK_API_KEY"):
        providers.append(DeepSeekProvider(deepseek_key, "deepseek-chat"))
        print("✅ DeepSeek configurado")
    else:
        print("⚠️ DEEPSEEK_API_KEY no encontrado")
    
    return providers

def main():
    """Ejecuta evaluación completa."""
    print("🚀 Iniciando evaluación del sistema RAG mejorado...")
    
    # Configura proveedores
    providers = setup_providers()
    if not providers:
        print("❌ No se encontraron proveedores configurados")
        return
    
    # Configura sistema de embeddings
    print("📚 Cargando sistema de embeddings...")
    embedding_system = EmbeddingSystem()
    
    # Verificar si existe el índice
    if not os.path.exists("data/index.faiss"):
        print("❌ Índice FAISS no encontrado. Ejecuta primero: python scripts/build_index.py")
        return
    
    embedding_system.load_index("data/index.faiss", "data/chunks.parquet")
    print("✅ Sistema de embeddings cargado")
    
    # Configura sistema RAG
    rag_system = RAGSystem(
        embedding_system=embedding_system,
        providers=providers,
        use_qdrant=False  # Usar FAISS por simplicidad
    )
    
    # Configura evaluador
    evaluator = RAGEvaluator(rag_system)
    
    # Ejecuta evaluación
    print("\n🔍 Ejecutando evaluación...")
    results = evaluator.run_full_evaluation(
        eval_file="eval/gold_questions.csv",
        reference_file="eval/reference_answers.csv"
    )
    
    # Guarda resultados
    output_file = "eval/evaluation_results_improved.json"
    evaluator.save_results(results, output_file)
    
    print(f"\n✅ Evaluación completada. Resultados guardados en {output_file}")
    
    # Muestra resumen de mejoras
    print("\n" + "="*60)
    print("RESUMEN DE MEJORAS IMPLEMENTADAS")
    print("="*60)
    print("✅ Sistema de evaluación corregido")
    print("✅ Dataset actualizado con fuentes reales")
    print("✅ Prompts mejorados para mayor precisión")
    print("✅ Evaluación semántica con respuestas de referencia")
    print("✅ Métricas de calidad multi-criterio")

if __name__ == "__main__":
    main()