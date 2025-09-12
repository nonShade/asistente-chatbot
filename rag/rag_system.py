import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scripts.ingest_documents import DocumentChunk
from rag.embedding_system import EmbeddingSystem
from providers.base import BaseLLMProvider


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

    SYSTEM_PROMPT = """Eres un asistente especializado en normativa y reglamentos de la Universidad de La Frontera (UFRO).

INSTRUCCIONES IMPORTANTES:
1. Analiza cuidadosamente toda la información proporcionada en el contexto
2. Si encuentras información relevante que responda a la pregunta, úsala y cita tus fuentes con el formato: [Nombre del documento, página X]
3. Si la información del contexto es parcialmente relevante pero no responde completamente la pregunta, proporciona lo que puedas basándote en el contexto y menciona qué información específica no está disponible
4. SOLO si absolutamente NINGUNA información del contexto es relevante para la pregunta, responde: "No encontré información sobre esto en la normativa UFRO disponible. Te sugiero contactar a [oficina correspondiente]"
5. Sé preciso y cita todas las fuentes relevantes
6. Usa un tono formal pero amigable
7. Si hay información contradictoria, menciona ambas fuentes y sus diferencias
8. Incluso si el contexto parece tangencialmente relacionado, trata de extraer cualquier información útil

CONTEXTO DE NORMATIVA UFRO:
{context}

PREGUNTA: {question}"""

    def __init__(self, embedding_system: EmbeddingSystem, providers: List[BaseLLMProvider]):
        self.embedding_system = embedding_system
        self.providers = providers

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

    def retrieve_context(self, query: str, k: int = 8) -> Tuple[str, List[Dict[str, Any]]]:
        """Recupera contexto relevante e información de fuentes con mejor cobertura."""
        # Mejora la consulta
        enhanced_query = self.rewrite_query(query)

        # Busca chunks relevantes con k más alto para mejor cobertura
        results = self.embedding_system.search(enhanced_query, k=k)

        if not results:
            return "", []

        # Construye contexto y fuentes
        context_parts = []
        sources = []

        for chunk, score in results:
            context_parts.append(f"[{chunk.title}, página {chunk.page}]: {chunk.content}")

            sources.append({
                'title': chunk.title,
                'page': chunk.page,
                'content': chunk.content[:300] + "...",  # Más contexto
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

        return provider.chat(messages, temperature=0.1, max_tokens=1500)

    def should_abstain(self, sources: List[Dict[str, Any]], query: str) -> Optional[str]:
        """Determina si el sistema debe abstenerse debido a la falta completa de fuentes."""
        # Solo se abstiene si absolutamente no hay fuentes encontradas
        if not sources:
            return "No encontré información sobre esto en la normativa UFRO disponible. Te sugiero contactar a la Dirección de Asuntos Estudiantiles o la Secretaría Académica."

        # Permite que el modelo decida si la información es relevante en lugar de usar un umbral de score
        # El modelo es mejor determinando relevancia semántica que un simple score de similitud
        return None

    def process_query(self, query: str, provider_name: Optional[str] = None) -> List[RAGResponse]:
        """Procesa consulta a través del pipeline RAG."""
        # Recupera contexto
        context, sources = self.retrieve_context(query)

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
            result = self.generate_response(query, context, provider)

            if 'error' in result:
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

        return responses

    def compare_providers(self, query: str) -> Dict[str, RAGResponse]:
        """Compara respuestas de todos los proveedores."""
        responses = self.process_query(query)
        return {response.provider_name: response for response in responses}
