import time
from typing import Dict, List, Any
from openai import OpenAI
from .base import BaseLLMProvider


class ChatGPTProvider(BaseLLMProvider):
    """Proveedor ChatGPT usando la API de OpenAI."""

    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # por 1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
    }

    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return f"ChatGPT ({self.model})"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Envía solicitud de chat completion a OpenAI."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1500)
            )

            latency = self._measure_latency(start_time)

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0
            cost = self.estimate_cost(input_tokens, output_tokens)

            return {
                "response": response.choices[0].message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency": latency,
                "cost": cost,
                "model": self.model
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency": self._measure_latency(start_time),
                "cost": 0.0
            }

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo basado en los precios actuales de OpenAI."""
        if self.model in self.PRICING:
            prices = self.PRICING[self.model]
            input_cost = (input_tokens / 1000) * prices["input"]
            output_cost = (output_tokens / 1000) * prices["output"]
            return input_cost + output_cost
        return 0.0
