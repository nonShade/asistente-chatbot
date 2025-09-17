import time
from typing import Dict, List, Any
import google.generativeai as genai
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Proveedor Gemini usando la API de Google."""

    PRICING = {
        "gemini-1.5-flash": {"input": 0.00015, "output": 0.0006},  # por 1K tokens
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Envía solicitud de chat completion a Gemini."""
        start_time = time.time()

        try:
            # Convertir mensajes al formato de Gemini
            conversation_text = self._format_messages(messages)

            # Configurar parámetros de generación
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.2),
                max_output_tokens=kwargs.get("max_tokens", 1500),
            )

            response = self.model_instance.generate_content(
                conversation_text, generation_config=generation_config
            )

            latency = self._measure_latency(start_time)

            # Estimar tokens (Gemini no siempre proporciona conteo exacto)
            input_tokens = len(conversation_text.split()) * 1.3  # Estimación aproximada
            output_tokens = len(response.text.split()) * 1.3 if response.text else 0
            total_tokens = input_tokens + output_tokens
            cost = self.estimate_cost(int(input_tokens), int(output_tokens))

            return {
                "response": response.text,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(total_tokens),
                "latency": latency,
                "cost": cost,
                "model": self.model,
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency": self._measure_latency(start_time),
                "cost": 0.0,
            }

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convierte mensajes de formato OpenAI a texto para Gemini."""
        formatted_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted_parts.append(f"Instrucciones del sistema: {content}")
            elif role == "user":
                formatted_parts.append(f"Usuario: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Asistente: {content}")

        return "\n\n".join(formatted_parts)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo basado en los precios actuales de Gemini."""
        if self.model in self.PRICING:
            prices = self.PRICING[self.model]
            input_cost = (input_tokens / 1000) * prices["input"]
            output_cost = (output_tokens / 1000) * prices["output"]
            return input_cost + output_cost
        return 0.0

