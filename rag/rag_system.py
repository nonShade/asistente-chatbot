import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scripts.ingest_documents import DocumentChunk
from rag.embedding_system import EmbeddingSystem
from providers.base import BaseLLMProvider

try:
    from rag.qdrant_client import UFROQdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    provider_name: str
    tokens_used: int
    latency: float
    cost: float


class RAGSystem:
    """Sistema RAG completo para asistente de normativa UFRO."""

    SYSTEM_PROMPT = """Eres un asistente especializado en normativa universitaria de la Universidad de La Frontera (UFRO). Tu objetivo es proporcionar respuestas precisas, completas y bien estructuradas.

INSTRUCCIONES ESPECÍFICAS:
1. ANALIZA cuidadosamente toda la información del contexto antes de responder
2. PROPORCIONA respuestas detalladas con información específica (fechas, requisitos, procesos paso a paso)
3. ESTRUCTURA tu respuesta usando viñetas o numeración cuando sea apropiado
4. CITA siempre las fuentes exactas usando el formato: [Nombre del documento, página X]
5. Si hay múltiples aspectos en la pregunta, abórdalos todos sistemáticamente

FORMATO DE RESPUESTA REQUERIDO:
- Respuesta completa y detallada (mínimo 100 palabras para preguntas complejas)
- Información específica extraída directamente del contexto
- Citas correctas al final de cada punto relevante
- Estructura clara con párrafos o listas según corresponda

MANEJO DE CASOS ESPECIALES:
- Si la información está parcialmente disponible: proporciona lo que tienes y especifica qué falta
- Si no hay información relevante: indica claramente que no se encontró información específica
- Para procesos complejos: explica paso a paso con todos los detalles disponibles

CONTEXTO DE DOCUMENTOS UFRO:
{context}

PREGUNTA DEL USUARIO: {question}

RESPUESTA DETALLADA Y BIEN DOCUMENTADA:"""

    def __init__(self, embedding_system: EmbeddingSystem, providers: List[BaseLLMProvider], 
                 use_qdrant: bool = False, qdrant_client=None):
        self.embedding_system = embedding_system
        self.providers = providers
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE
        self.qdrant_client = qdrant_client
        
        if use_qdrant and not QDRANT_AVAILABLE:
            print("⚠️ Qdrant no disponible, usando FAISS")
        elif use_qdrant and qdrant_client is None:
            print("⚠️ Cliente Qdrant no proporcionado, usando FAISS")

    def rewrite_query(self, query: str) -> str:
        """Reescribe la consulta para mejor recuperación (mejora opcional)."""
        # Mejora simple de consulta - podría mejorar con LLM
        enhanced = query.lower().strip()

        # Agrega palabras clave de contexto para mejor coincidencia
        keywords = {
            'matrícula': 'matricula inscripción',
            'titulación': 'titulacion graduación tesis',
            'apelación': 'apelacion recurso reclamación',
            'beneficios': 'beneficios becas ayudas',
            'calendario': 'calendario fechas académico'
        }

        for key, expansion in keywords.items():
            if key in enhanced:
                enhanced += f" {expansion}"

        return enhanced

    def retrieve_context(self, query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Recupera contexto relevante e información de fuentes con mejor cobertura."""
        # Mejora la consulta
        enhanced_query = self.rewrite_query(query)

        if self.use_qdrant and self.qdrant_client:
            # Usar Qdrant para búsqueda
            query_embedding = self.embedding_system.embed_texts([enhanced_query])[0]
            results = self.qdrant_client.search_similar(query_embedding, limit=k)
            
            if not results:
                return "", []
            
            # Construye contexto y fuentes para Qdrant
            context_parts = []
            sources = []
            
            for result in results:
                context_parts.append(f"[{result['title']}, página {result['page']}]: {result['text']}")
                
                sources.append({
                    'title': result['title'],
                    'page': result['page'],
                    'content': result['text'][:300] + "...",
                    'doc_id': result['doc_id'],
                    'score': result['score'],
                    'url': result['url'],
                    'vigencia': result['vigencia']
                })
        else:
            # Usar FAISS para búsqueda (comportamiento original)
            results = self.embedding_system.search(enhanced_query, k=k)

            if not results:
                return "", []

            # Construye contexto y fuentes para FAISS
            context_parts = []
            sources = []

            for chunk, score in results:
                context_parts.append(f"[{chunk.title}, página {chunk.page}]: {chunk.content}")

                sources.append({
                    'title': chunk.title,
                    'page': chunk.page,
                    'content': chunk.content[:300] + "...",
                    'doc_id': chunk.doc_id,
                    'score': score,
                    'url': chunk.url,
                    'vigencia': chunk.vigencia
                })

        context = "\n\n".join(context_parts)
        return context, sources

    def generate_response(self, query: str, context: str, provider: BaseLLMProvider) -> Dict[str, Any]:
        """Genera respuesta usando el proveedor especificado."""
        prompt = self.SYSTEM_PROMPT.format(context=context, question=query)

        messages = [
            {"role": "system", "content": "Eres un asistente especializado en normativa universitaria."},
            {"role": "user", "content": prompt}
        ]

        return provider.chat(messages, temperature=0.3, max_tokens=1200)

    def should_abstain(self, sources: List[Dict[str, Any]], query: str) -> Optional[str]:
        """Determina si el sistema debe abstenerse debido a la falta completa de fuentes."""
        # Solo se abstiene si absolutamente no hay fuentes encontradas
        if not sources:
            return "No encontré información sobre esto en la normativa UFRO disponible. Te sugiero contactar a la Dirección de Asuntos Estudiantiles o la Secretaría Académica."

        # Permite que el modelo decida si la información es relevante en lugar de usar un umbral de score
        # El modelo es mejor determinando relevancia semántica que un simple score de similitud
        return None

    def process_query(self, query: str, provider_name: Optional[str] = None, k: int = 5) -> List[RAGResponse]:
        """Procesa consulta a través del pipeline RAG."""
        # Recupera contexto
        context, sources = self.retrieve_context(query, k)

        # Verifica si debe abstenerse
        abstention = self.should_abstain(sources, query)
        if abstention:
            return [RAGResponse(
                answer=abstention,
                sources=[],
                provider_name="System",
                tokens_used=0,
                latency=0.0,
                cost=0.0
            )]

        responses = []

        # Genera respuestas del/los proveedor(es) especificado(s)
        providers_to_use = self.providers
        if provider_name:
            providers_to_use = [p for p in self.providers if provider_name.lower() in p.name.lower()]

        for provider in providers_to_use:
            try:
                result = self.generate_response(query, context, provider)
                
                if 'error' in result:
                    print(f"❌ Error en {provider.name}: {result['error']}")
                    continue

                response = RAGResponse(
                    answer=result['response'],
                    sources=sources,
                    provider_name=provider.name,
                    tokens_used=result.get('total_tokens', 0),
                    latency=result.get('latency', 0.0),
                    cost=result.get('cost', 0.0)
                )
                responses.append(response)
                
            except Exception as e:
                print(f"❌ Error inesperado con {provider.name}: {str(e)}")
                continue

        return responses

    def compare_providers(self, query: str) -> Dict[str, RAGResponse]:
        """Compara respuestas de todos los proveedores."""
        responses = self.process_query(query)
        return {response.provider_name: response for response in responses}
