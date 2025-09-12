import os
import requests
from bs4 import BeautifulSoup
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno desde .env
load_dotenv()

class HTMLToDeepSeek:
    def __init__(self, deepseek_model: str = "deepseek-reasoner", base_url: str = "https://api.deepseek.com", api_key: Optional[str] = None):
        """
        Clase para convertir HTML a formato entendible y enviarlo a DeepSeek.
        """
        self.model = deepseek_model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError("API key para DeepSeek no encontrada en entorno o parámetro.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def fetch_html(self, url: str) -> str:
        """
        Obtiene el contenido HTML de una URL.
        """
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def clean_html(self, html: str, selector: Optional[str] = None) -> str:
        """
        Si se proporciona un selector CSS, extrae solo esa sección.
        Si no, devuelve todo el HTML formateado.
        """
        soup = BeautifulSoup(html, 'html.parser')

        if selector:
            target = soup.select_one(selector)
            if not target:
                raise ValueError(f"No se encontró contenido para el selector: {selector}")
            return target.prettify()
        return soup.prettify()

    def send_to_deepseek(self, content: str, system_prompt: Optional[str] = None) -> str:
        """
        Envía el contenido a DeepSeek y devuelve la respuesta.
        """
        messages = []
        system_msg = system_prompt or "Eres un asistente especializado en análisis de calendarios académicos universitarios. Analiza el contenido HTML del calendario y extrae información relevante sobre fechas importantes, períodos académicos, inscripciones, exámenes y eventos especiales."
        messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False
        )
        
        # Asegurar que retornamos una cadena no nula
        result = response.choices[0].message.content
        return result if result is not None else "No se recibió respuesta del modelo."

    def save_calendar_info(self, content: str, output_file: str = "data/raw/calendario_academico_2025.txt") -> str:
        """
        Guarda la información del calendario procesada en un archivo de texto.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Agregar metadatos al contenido
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""CALENDARIO ACADÉMICO UFRO 2025
Fuente: https://www.ufro.cl/calendario-academico/
Procesado: {timestamp}
Procesado por: DeepSeek API

---

"""
        
        full_content = header + content
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"Información del calendario guardada en: {output_file}")
        return output_file

    def process_url(self, url: str, selector: Optional[str] = None, system_prompt: Optional[str] = None, save_to_file: bool = True) -> str:
        """
        Flujo completo: descarga -> limpia -> envía a DeepSeek -> retorna respuesta.
        Si save_to_file es True, también guarda la información en un archivo.
        """
        raw_html = self.fetch_html(url)
        clean = self.clean_html(raw_html, selector)
        result = self.send_to_deepseek(clean, system_prompt)
        
        if save_to_file:
            self.save_calendar_info(result)
        
        return result

if __name__ == "__main__":
   URL = "https://www.ufro.cl/calendario-academico/"
   SELECTOR = "#content"

   processor = HTMLToDeepSeek()
   print(processor.process_url(URL, selector=SELECTOR))