import os
import re
import pandas as pd
from typing import List, Dict, Tuple
from pypdf import PdfReader
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    doc_id: str
    title: str
    content: str
    page: int
    chunk_id: str
    url: str
    vigencia: str


class DocumentIngester:
    """Maneja la ingestión y fragmentación de documentos PDF."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_file(self, file_path: str) -> List[Tuple[str, int]]:
        """Extrae texto de un archivo, soporta PDF y TXT."""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            return self.extract_text_from_txt(file_path)
        else:
            print(f"Formato de archivo no soportado: {file_path}")
            return []

    def extract_text_from_txt(self, txt_path: str) -> List[Tuple[str, int]]:
        """Extrae texto de un archivo TXT, tratando cada párrafo como una 'página'."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Dividir por párrafos vacíos (doble salto de línea)
            sections = content.split('\n\n')
            pages_text = []
            
            for page_num, section in enumerate(sections, 1):
                if section.strip():
                    cleaned_text = self._clean_text(section)
                    pages_text.append((cleaned_text, page_num))
            
            return pages_text
            
        except Exception as e:
            print(f"Error procesando {txt_path}: {str(e)}")
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extrae texto de un PDF, retornando tuplas (texto, número de página)."""
        try:
            reader = PdfReader(pdf_path)
            pages_text = []

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Limpiar texto: normalizar espacios y eliminar encabezados/pies de página
                    cleaned_text = self._clean_text(text)
                    pages_text.append((cleaned_text, page_num))

            return pages_text

        except Exception as e:
            print(f"Error procesando {pdf_path}: {str(e)}")
            return []

    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído con un mejor procesamiento."""
        # Eliminar espacios en blanco excesivos
        text = re.sub(r'\s+', ' ', text)

        # Eliminar patrones comunes de encabezado/pie de página (específicos para UFRO)
        text = re.sub(r'Página \d+', '', text)
        text = re.sub(r'\d+/\d+', '', text)
        text = re.sub(r'Universidad de La Frontera.*?UFRO', '', text, flags=re.IGNORECASE)
        text = re.sub(r'www\.ufro\.cl', '', text, flags=re.IGNORECASE)

        # Preservar estructura de listas y numeración
        text = re.sub(r'(\d+\.)\s*', r'\1 ', text)  # Normalizar numeración
        text = re.sub(r'([a-z])\.\s*([A-Z])', r'\1. \2', text)  # Separar oraciones

        # Mantener artículos y secciones
        text = re.sub(r'(Art[íi]culo|Secci[óo]n|Cap[íi]tulo)\s*(\d+)', r'\1 \2:', text, flags=re.IGNORECASE)

        # Limpiar caracteres extraños pero mantener tildes y ñ
        text = re.sub(r'[^\w\s.,;:()\-ñáéíóúüÑÁÉÍÓÚÜ]', ' ', text)

        return text.strip()

    def chunk_text(self, text: str, doc_id: str, page: int) -> List[str]:
        """Divide el texto en fragmentos superpuestos con mejor detección de límites."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size


            # Intentar cortar en límites más naturales
            if end < len(text):
                chunk_text = text[start:end]

                # Orden de prioridad para los puntos de corte
                break_points = [
                    chunk_text.rfind('. '),
                    chunk_text.rfind('? '),
                    chunk_text.rfind('! '),
                    chunk_text.rfind(': '),
                    chunk_text.rfind('; '),
                    chunk_text.rfind(', '),
                    chunk_text.rfind(' ')
                ]

                best_break = -1
                for bp in break_points:
                    if bp > self.chunk_size * 0.6:  # No cortar demasiado pronto
                        best_break = bp + 1
                        break

                if best_break > 0:
                    end = start + best_break

            chunk = text[start:end].strip()

            if chunk and len(chunk) > 50:  # Omitir fragmentos muy cortos
                chunks.append(chunk)


            # Mover la posición inicial considerando el solapamiento
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def process_documents(self, data_dir: str, sources_file: str) -> List[DocumentChunk]:
        """Procesa todos los documentos y retorna fragmentos con metadatos."""
        sources_df = pd.read_csv(sources_file)
        all_chunks = []

        for _, row in sources_df.iterrows():
            file_path = os.path.join(data_dir, str(row['filename']))

            if not os.path.exists(file_path):
                print(f"Advertencia: Archivo no encontrado: {file_path}")
                continue

            print(f"Procesando: {row['title']}")

            # Extraer texto del archivo (PDF o TXT)
            pages_text = self.extract_text_from_file(file_path)

            # Procesar cada página
            for page_text, page_num in pages_text:
                # Crear fragmentos para esta página
                chunks = self.chunk_text(page_text, str(row['doc_id']), page_num)

                # Crear objetos DocumentChunk
                for chunk_idx, chunk_content in enumerate(chunks):
                    chunk = DocumentChunk(
                        doc_id=str(row['doc_id']),
                        title=str(row['title']),
                        content=chunk_content,
                        page=page_num,
                        chunk_id=f"{row['doc_id']}_p{page_num}_c{chunk_idx}",
                        url=str(row['url']),
                        vigencia=str(row['vigencia'])
                    )
                    all_chunks.append(chunk)

        return all_chunks

    def save_chunks(self, chunks: List[DocumentChunk], output_file: str):
        """Guarda los fragmentos en un archivo Parquet."""
        data = []
        for chunk in chunks:
            data.append({
                'doc_id': chunk.doc_id,
                'title': chunk.title,
                'content': chunk.content,
                'page': chunk.page,
                'chunk_id': chunk.chunk_id,
                'url': chunk.url,
                'vigencia': chunk.vigencia
            })

        df = pd.DataFrame(data)
        df.to_parquet(output_file, index=False)
        print(f"Se guardaron {len(chunks)} fragmentos en {output_file}")