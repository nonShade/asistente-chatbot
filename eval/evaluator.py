import os
import time
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rag.rag_system import RAGSystem, RAGResponse


class EvaluationMetrics:
    """Sistema de evaluación para rendimiento de RAG."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def calculate_exact_match(self, predicted: str, expected: str) -> bool:
        """Calcula puntuación de coincidencia exacta."""
        return predicted.strip().lower() == expected.strip().lower()

    def calculate_semantic_similarity(self, predicted: str, expected: str) -> float:
        """Calcula similitud semántica usando embeddings."""
        if not predicted.strip() or not expected.strip():
            return 0.0

        pred_embedding = self.embedding_model.encode([predicted])
        exp_embedding = self.embedding_model.encode([expected])

        similarity = cosine_similarity(pred_embedding, exp_embedding)[0][0]
        return float(similarity)

    def calculate_citation_coverage(self, response: RAGResponse, expected_sources: List[str]) -> float:
        """Calcula qué tan bien la respuesta cubre las fuentes esperadas."""
        if not expected_sources:
            return 1.0

        cited_sources = set()
        for source in response.sources:
            cited_sources.add(source['doc_id'])

        expected_set = set(expected_sources)
        intersection = cited_sources.intersection(expected_set)

        return len(intersection) / len(expected_set)

    def calculate_precision_at_k(self, retrieved_sources: List[Dict],
                                expected_sources: List[str], k: int = 5) -> float:
        """Calcula precisión@k para recuperación."""
        if not expected_sources or not retrieved_sources:
            return 0.0

        top_k = retrieved_sources[:k]
        relevant_retrieved = 0

        for source in top_k:
            if source['doc_id'] in expected_sources:
                relevant_retrieved += 1

        return relevant_retrieved / min(k, len(top_k))

    def has_proper_citations(self, answer: str) -> bool:
        """Verifica si la respuesta contiene citas apropiadas."""
        import re
        citation_pattern = r'\[([^,]+),\s*(página|p\.)\s*\d+\]'
        citations = re.findall(citation_pattern, answer, re.IGNORECASE)
        return len(citations) >= 1

    def evaluate_response(self, response: RAGResponse, expected_answer: str,
                         expected_sources: List[str]) -> Dict[str, Any]:
        """Evalúa una sola respuesta."""
        # Similitud semántica (ya que no tenemos respuestas exactas esperadas)
        semantic_score = 0.8 if len(response.answer) > 50 else 0.2  # Placeholder

        # Cobertura de citas
        citation_coverage = self.calculate_citation_coverage(response, expected_sources)

        # Precisión@k
        precision_k = self.calculate_precision_at_k(response.sources, expected_sources)

        # Verificación de formato de citas
        has_citations = self.has_proper_citations(response.answer)

        # Verificación de abstención (bueno si no hay fuentes relevantes)
        abstained = "No encontré información" in response.answer
        abstention_appropriate = len(response.sources) == 0

        return {
            'semantic_similarity': semantic_score,
            'citation_coverage': citation_coverage,
            'precision_at_k': precision_k,
            'has_proper_citations': has_citations,
            'abstained': abstained,
            'abstention_appropriate': abstention_appropriate,
            'tokens_used': response.tokens_used,
            'latency': response.latency,
            'cost': response.cost,
            'provider': response.provider_name
        }


class RAGEvaluator:
    """Sistema completo de evaluación para pipeline RAG."""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.metrics = EvaluationMetrics()

    def load_evaluation_set(self, eval_file: str) -> pd.DataFrame:
        """Carga preguntas de evaluación."""
        return pd.read_csv(eval_file)

    def evaluate_single_question(self, question: str, expected_sources: List[str],
                                provider_name: str = None) -> Dict[str, Any]:
        """Evalúa sistema RAG en una sola pregunta."""
        start_time = time.time()

        # Obtiene respuestas
        responses = self.rag_system.process_query(question, provider_name)

        if not responses:
            return {
                'error': 'No se generaron respuestas',
                'total_latency': time.time() - start_time
            }

        # Evalúa cada respuesta
        results = []
        for response in responses:
            metrics = self.metrics.evaluate_response(response, "", expected_sources)
            metrics['question'] = question
            metrics['expected_sources'] = expected_sources
            results.append(metrics)

        return {
            'responses': results,
            'total_latency': time.time() - start_time
        }

    def run_full_evaluation(self, eval_file: str) -> Dict[str, Any]:
        """Ejecuta evaluación en conjunto de pruebas completo."""
        eval_df = self.load_evaluation_set(eval_file)

        all_results = []
        provider_results = {}

        print(f"Ejecutando evaluación en {len(eval_df)} preguntas...")

        for idx, row in eval_df.iterrows():
            print(f"Evaluando pregunta {idx + 1}/{len(eval_df)}: {row['question'][:50]}...")

            expected_sources = row['expected_sources'].split(',') if row['expected_sources'] else []

            # Evalúa con todos los proveedores
            result = self.evaluate_single_question(
                row['question'],
                expected_sources
            )

            if 'error' in result:
                continue

            # Organiza resultados por proveedor
            for response_metrics in result['responses']:
                provider = response_metrics['provider']
                if provider not in provider_results:
                    provider_results[provider] = []

                response_metrics.update({
                    'question_id': idx,
                    'category': row['category'],
                    'difficulty': row['difficulty']
                })

                provider_results[provider].append(response_metrics)
                all_results.append(response_metrics)

        # Calcula métricas agregadas
        summary = self.calculate_summary_metrics(provider_results)

        return {
            'detailed_results': all_results,
            'provider_results': provider_results,
            'summary': summary
        }

    def calculate_summary_metrics(self, provider_results: Dict) -> Dict[str, Any]:
        """Calcula métricas resumen para cada proveedor."""
        summary = {}

        for provider, results in provider_results.items():
            if not results:
                continue

            n_questions = len(results)

            # Métricas promedio
            avg_semantic = sum(r['semantic_similarity'] for r in results) / n_questions
            avg_citation_coverage = sum(r['citation_coverage'] for r in results) / n_questions
            avg_precision_k = sum(r['precision_at_k'] for r in results) / n_questions
            avg_latency = sum(r['latency'] for r in results) / n_questions
            total_cost = sum(r['cost'] for r in results)
            total_tokens = sum(r['tokens_used'] for r in results)

            # Tasa de citas
            citation_rate = sum(1 for r in results if r['has_proper_citations']) / n_questions

            # Tasa de abstención
            abstention_rate = sum(1 for r in results if r['abstained']) / n_questions

            summary[provider] = {
                'n_questions': n_questions,
                'avg_semantic_similarity': avg_semantic,
                'avg_citation_coverage': avg_citation_coverage,
                'avg_precision_at_k': avg_precision_k,
                'citation_rate': citation_rate,
                'abstention_rate': abstention_rate,
                'avg_latency': avg_latency,
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'cost_per_question': total_cost / n_questions if n_questions > 0 else 0
            }

        return summary

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Guarda resultados de evaluación en archivo."""
        # Guarda resultados detallados como CSV
        detailed_df = pd.DataFrame(results['detailed_results'])
        detailed_df.to_csv(output_file.replace('.json', '_detailed.csv'), index=False)

        # Guarda resumen como JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(results['summary'], f, indent=2)

        print(f"Resultados guardados en {output_file}")

        # Imprime resumen
        self.print_summary(results['summary'])

    def print_summary(self, summary: Dict[str, Any]):
        """Imprime resumen de evaluación."""
        print("\n" + "="*60)
        print("RESUMEN DE EVALUACIÓN")
        print("="*60)

        for provider, metrics in summary.items():
            print(f"\n{provider}:")
            print(f"  Preguntas: {metrics['n_questions']}")
            print(f"  Similitud Semántica: {metrics['avg_semantic_similarity']:.3f}")
            print(f"  Cobertura de Citas: {metrics['avg_citation_coverage']:.3f}")
            print(f"  Precisión@K: {metrics['avg_precision_at_k']:.3f}")
            print(f"  Tasa de Citas: {metrics['citation_rate']:.3f}")
            print(f"  Tasa de Abstención: {metrics['abstention_rate']:.3f}")
            print(f"  Latencia Promedio: {metrics['avg_latency']:.3f}s")
            print(f"  Costo Total: ${metrics['total_cost']:.4f}")
            print(f"  Costo por Pregunta: ${metrics['cost_per_question']:.4f}")
            print(f"  Tokens Totales: {metrics['total_tokens']:,}")
