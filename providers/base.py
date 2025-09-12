from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time


class BaseLLMProvider(ABC):
    """Clase base para proveedores de LLM siguiendo el patrón adaptador."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    @abstractmethod
    def name(self) -> str:
        """Retorna el nombre del proveedor."""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Envía solicitud de chat completion y retorna respuesta con metadatos.

        Args:
            messages: Lista de diccionarios de mensajes con 'role' y 'content'
            **kwargs: Parámetros adicionales (temperature, max_tokens, etc.)

        Returns:
            Diccionario conteniendo respuesta, tokens usados, latencia y estimación de costo
        """
        pass

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo en USD basado en el uso de tokens."""
        return 0.0

    def _measure_latency(self, start_time: float) -> float:
        """Función auxiliar para medir latencia de solicitud."""
        return time.time() - start_time
