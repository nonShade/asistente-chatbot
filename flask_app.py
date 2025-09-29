#!/usr/bin/env python3
"""
Flask Web App for UFRO Chatbot
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import uuid
import logging
from datetime import datetime
from typing import Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agrega la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from providers.deepseek import DeepSeekProvider
from providers.chatgpt import ChatGPTProvider
from rag.embedding_system import EmbeddingSystem
from rag.rag_system import RAGSystem

try:
    from rag.qdrant_client import UFROQdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

class UFROFlaskApp:
    """Aplicación Flask para el chatbot UFRO."""
    
    def __init__(self):
        load_dotenv()
        self.providers = []
        self.rag_system = None
        self.embedding_system = None
        self.qdrant_client = None
        self.use_qdrant = os.getenv('USE_QDRANT', 'false').lower() == 'true'
        self.setup_providers()
        self.setup_rag_system()

    def setup_providers(self):
        """Inicializa los proveedores de modelos de lenguaje (LLM)."""
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')

        if deepseek_key:
            deepseek_model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
            self.providers.append(DeepSeekProvider(deepseek_key, deepseek_model))
            logger.info(f"✓ Proveedor DeepSeek inicializado ({deepseek_model})")
        else:
            logger.warning("⚠ No se encontró DEEPSEEK_API_KEY")

        if openai_key:
            openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
            self.providers.append(ChatGPTProvider(openai_key, openai_model))
            logger.info(f"✓ Proveedor ChatGPT inicializado ({openai_model})")
        else:
            logger.warning("⚠ No se encontró OPENAI_API_KEY")

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
                    logger.info("✓ Conectado a Qdrant")
                    self.rag_system = RAGSystem(self.embedding_system, self.providers, 
                                              use_qdrant=True, qdrant_client=self.qdrant_client)
                    logger.info("✓ Sistema RAG inicializado con Qdrant")
                    return
                else:
                    logger.warning("⚠️ Qdrant no disponible, usando FAISS")
                    self.use_qdrant = False
            except Exception as e:
                logger.error(f"⚠️ Error conectando a Qdrant: {e}")
                logger.warning("⚠️ Usando FAISS como respaldo")
                self.use_qdrant = False

        # Usar FAISS (comportamiento original)
        index_path = 'data/index.faiss'
        chunks_path = 'data/chunks_improved.parquet'

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            logger.info("📚 Cargando índice FAISS existente...")
            self.embedding_system.load_index(index_path, chunks_path)
        else:
            logger.warning("⚠️ No se encontraron índices FAISS. Ejecuta el script de construcción primero.")

        self.rag_system = RAGSystem(self.embedding_system, self.providers)
        logger.info("✓ Sistema RAG inicializado con FAISS")

    def process_query(self, query: str, provider_name: Optional[str] = None, k: int = 5):
        """Procesa una consulta y devuelve la respuesta."""
        if not self.rag_system:
            return {"error": "Sistema RAG no inicializado"}
        
        try:
            responses = self.rag_system.process_query(query, provider_name, k)
            
            if not responses:
                return {"error": "No se obtuvieron respuestas del sistema RAG"}
            
            # Formatear respuesta para web
            response = responses[0]  # Tomar la primera respuesta
            
            return {
                "answer": response.answer,
                "sources": response.sources[:3],  # Limitar a 3 fuentes
                "provider": response.provider_name,
                "metrics": {
                    "tokens": response.tokens_used,
                    "latency": round(response.latency, 2),
                    "cost": round(response.cost, 4)
                }
            }
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {str(e)}")
            return {"error": f"Error procesando consulta: {str(e)}"}

# Inicializar la aplicación
chatbot_app = UFROFlaskApp()

@app.route('/')
def index():
    """Página principal del chatbot."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/health')
def health():
    """Endpoint de salud."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "providers": len(chatbot_app.providers),
        "rag_initialized": chatbot_app.rag_system is not None
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint para procesar mensajes del chat."""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Mensaje requerido"}), 400
        
        query = data['message'].strip()
        if not query:
            return jsonify({"error": "Mensaje vacío"}), 400
        
        provider = data.get('provider')
        k = data.get('k', 5)
        
        # Validar parámetros
        if provider and provider not in ['deepseek', 'chatgpt']:
            return jsonify({"error": "Proveedor no válido"}), 400
        
        if not isinstance(k, int) or k < 1 or k > 10:
            return jsonify({"error": "k debe ser un entero entre 1 y 10"}), 400
        
        # Procesar consulta
        result = chatbot_app.process_query(query, provider, k)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify({
            "success": True,
            "result": result,
            "session_id": session.get('session_id')
        })
        
    except Exception as e:
        logger.error(f"Error en /api/chat: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/api/providers')
def get_providers():
    """Obtiene la lista de proveedores disponibles."""
    return jsonify({
        "providers": [provider.name.lower().replace(' ', '') for provider in chatbot_app.providers]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Iniciando servidor Flask en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)