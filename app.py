#!/usr/bin/env python3
"""
UFRO Chatbot - Asistente de normativa universitaria
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from typing import Optional

 # Agrega la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from providers.deepseek import DeepSeekProvider
from providers.chatgpt import ChatGPTProvider
from rag.embedding_system import EmbeddingSystem
from rag.rag_system import RAGSystem
from scripts.ingest_documents import DocumentIngester
from eval.evaluator import RAGEvaluator

try:
    from rag.qdrant_client import UFROQdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class UFROChatbot:
    """Aplicación principal del chatbot."""

    def __init__(self):
        load_dotenv()
        self.providers = []
        self.rag_system = None
        self.embedding_system = None
        self.qdrant_client = None
        self.use_qdrant = os.getenv('USE_QDRANT', 'false').lower() == 'true'

    def setup_providers(self):
        """Inicializa los proveedores de modelos de lenguaje (LLM)."""
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')

        if deepseek_key:
            deepseek_model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
            self.providers.append(DeepSeekProvider(deepseek_key, deepseek_model))
            print(f"✓ Proveedor DeepSeek inicializado ({deepseek_model})")
        else:
            print("⚠ No se encontró DEEPSEEK_API_KEY")

        if openai_key:
            openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
            self.providers.append(ChatGPTProvider(openai_key, openai_model))
            print(f"✓ Proveedor ChatGPT inicializado ({openai_model})")
        else:
            print("⚠ No se encontró OPENAI_API_KEY")

        if not self.providers:
            raise ValueError("No hay claves API configuradas. Por favor revisa tu archivo .env.")

    def setup_rag_system(self):
        """Inicializa el sistema RAG."""
        embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_system = EmbeddingSystem(embedding_model)

        # Configurar Qdrant si está habilitado
        if self.use_qdrant and QDRANT_AVAILABLE:
            try:
                qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
                qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
                from rag.qdrant_client import UFROQdrantClient
                self.qdrant_client = UFROQdrantClient(host=qdrant_host, port=qdrant_port)
                
                if self.qdrant_client.health_check():
                    print("✓ Conectado a Qdrant")
                    self.rag_system = RAGSystem(self.embedding_system, self.providers, 
                                              use_qdrant=True, qdrant_client=self.qdrant_client)
                    print("✓ Sistema RAG inicializado con Qdrant")
                    return
                else:
                    print("⚠️ Qdrant no disponible, usando FAISS")
                    self.use_qdrant = False
            except Exception as e:
                print(f"⚠️ Error conectando a Qdrant: {e}")
                print("⚠️ Usando FAISS como respaldo")
                self.use_qdrant = False

        # Usar FAISS (comportamiento original)
        index_path = 'data/processed/index.faiss'
        chunks_path = 'data/processed/chunks.parquet'

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print("📚 Cargando índice FAISS existente...")
            self.embedding_system.load_index(index_path, chunks_path)
        else:
            print("🔄 Construyendo índice FAISS desde los documentos...")
            self.build_index()

        self.rag_system = RAGSystem(self.embedding_system, self.providers)
        print("✓ Sistema RAG inicializado con FAISS")

    def build_index(self):
        """Construye el índice FAISS a partir de los documentos."""
        chunk_size = int(os.getenv('CHUNK_SIZE', '900'))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '120'))

        ingester = DocumentIngester(chunk_size, chunk_overlap)

        print("📄 Procesando documentos...")
        chunks = ingester.process_documents('data/raw', 'data/sources.csv')

        print("🔍 Generando embeddings y construyendo el índice FAISS...")
        # Inicializar embedding_system si no existe
        if self.embedding_system is None:
            from rag.embedding_system import EmbeddingSystem
            self.embedding_system = EmbeddingSystem()
            
        self.embedding_system.build_and_save_index(
            chunks,
            'data/processed/index.faiss',
            'data/processed/chunks.parquet'
        )

    def interactive_mode(self):
        """Ejecuta el modo interactivo por línea de comandos (CLI)."""
        print("\n" + "="*60)
        print("🤖 ASISTENTE UFRO - Consultas sobre normativa universitaria")
        print("="*60)
        print("Escribe 'salir' para terminar, 'help' para ayuda")
        print("Comandos especiales:")
        print("  /compare <pregunta>  - Comparar respuestas de ambos proveedores")
        print("  /deepseek <pregunta> - Usar solo DeepSeek")
        print("  /chatgpt <pregunta>  - Usar solo ChatGPT")
        print("="*60)

        while True:
            try:
                query = input("\n💬 Tu pregunta: ").strip()

                if query.lower() in ['salir', 'exit', 'quit']:
                    print("¡Hasta luego! 👋")
                    break

                if query.lower() in ['help', 'ayuda']:
                    self._show_help()
                    continue

                if not query:
                    continue

                # Manejar comandos especiales
                provider_name = None
                compare_mode = False

                if query.startswith('/compare '):
                    query = query[9:]
                    compare_mode = True
                elif query.startswith('/deepseek '):
                    query = query[10:]
                    provider_name = 'deepseek'
                elif query.startswith('/chatgpt '):
                    query = query[9:]
                    provider_name = 'chatgpt'

                print("\n🔍 Buscando en la normativa UFRO...")

                if compare_mode:
                    self._handle_compare_mode(query)
                else:
                    self._handle_single_query(query, provider_name)

            except KeyboardInterrupt:
                print("\n\n¡Hasta luego! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    def _show_help(self):
        """Muestra la información de ayuda."""
        print("\n📖 AYUDA:")
        print("- Puedes hacer preguntas sobre normativa UFRO")
        print("- El sistema busca en reglamentos, calendarios y procedimientos")
        print("- Las respuestas incluyen citas de las fuentes oficiales")
        print("- Usa comandos especiales para comparar proveedores o elegir uno específico")

    def _handle_single_query(self, query: str, provider_name: Optional[str]):
        """Gestiona una consulta con selección opcional de proveedor."""
        try:
            if self.rag_system is None:
                print("❌ Sistema RAG no inicializado")
                return
            
            responses = self.rag_system.process_query(query, provider_name)
            
            if not responses:
                print("❌ No se obtuvieron respuestas del sistema RAG")
                return

            for response in responses:
                print(f"\n📋 Respuesta ({response.provider_name}):")
                print("-" * 50)
                print(response.answer)

                if response.sources:
                    print(f"\n📚 Fuentes consultadas ({len(response.sources)}):")
                    for i, source in enumerate(response.sources[:3], 1):
                        print(f"  {i}. {source['title']} (página {source['page']}) - Score: {source['score']:.3f}")

                print(f"\n📊 Métricas:")
                print(f"  Tokens: {response.tokens_used}")
                print(f"  Latencia: {response.latency:.2f}s")
                print(f"  Costo estimado: ${response.cost:.4f}")
                
        except Exception as e:
            print(f"❌ Error procesando consulta: {str(e)}")
            import traceback
            traceback.print_exc()

    def _handle_compare_mode(self, query: str):
        """Gestiona la comparación entre proveedores."""
        if self.rag_system is None:
            print("❌ Sistema RAG no inicializado")
            return
            
        provider_responses = self.rag_system.compare_providers(query)

        print(f"\n📊 Comparación de respuestas:")
        print("=" * 60)

        for provider_name, response in provider_responses.items():
            print(f"\n🤖 {provider_name}:")
            print("-" * 30)
            print(response.answer)
            print(f"Tokens: {response.tokens_used} | Latencia: {response.latency:.2f}s | Costo: ${response.cost:.4f}")

    def batch_evaluation(self, eval_file: str):
        """Ejecuta la evaluación batch."""
        print(f"🧪 Ejecutando evaluación batch desde {eval_file}")

        if self.rag_system is None:
            print("❌ Sistema RAG no inicializado")
            return None
            
        evaluator = RAGEvaluator(self.rag_system)
        results = evaluator.run_full_evaluation(eval_file)

        # Save results
        output_file = eval_file.replace('.csv', '_results.json')
        evaluator.save_results(results, output_file)

        return results


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description='UFRO Chatbot - Asistente de normativa')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'build-index'],
                       default='interactive', help='Modo de ejecución')
    parser.add_argument('--eval-file', default='eval/gold_questions.csv',
                       help='Archivo de evaluación para modo batch')

    args = parser.parse_args()

    try:
        chatbot = UFROChatbot()

        if args.mode == 'build-index':
            print("🔨 Construyendo índice FAISS...")
            chatbot.build_index()

        elif args.mode == 'batch':
            chatbot.setup_providers()
            chatbot.setup_rag_system()
            chatbot.batch_evaluation(args.eval_file)

        else:  # interactive
            chatbot.setup_providers()
            chatbot.setup_rag_system()
            chatbot.interactive_mode()

    except Exception as e:
        print(f"❌ Error fatal: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
