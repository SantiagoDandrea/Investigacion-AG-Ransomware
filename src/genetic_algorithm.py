"""
Módulo genetic_algorithm.py
Responsabilidad: Implementación del Algoritmo Genético para selección de características.
Incluye población binaria, selección por torneo (k=3), cruce de un punto (Pc=0.8),
mutación bit-flip (Pm=1/(2.5*N_total)), elitismo (1 individuo), reemplazo generacional,
criterios de parada (50 generaciones o 10 estancadas) y registro histórico.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.fitness import FitnessEvaluator


class GeneticAlgorithm:
    """
    Algoritmo Genético para selección de características en PE Headers.
    """
    def __init__(
        self,
        n_total: int,
        pop_size: int = 50,
        max_generations: int = 50,
        stagnation_limit: int = 10,
        tournament_size: int = 3,
        crossover_prob: float = 0.8,
        random_state: int = 42
    ):
        """
        Parámetros:
            n_total (int): Cantidad de características tras preprocesamiento (dinámico).
            pop_size (int): Tamaño de la población (fijo: 50).
            max_generations (int): Cantidad máxima de generaciones (fijo: 50).
            stagnation_limit (int): Límite de generaciones consecutivas sin mejora (fijo: 10).
            tournament_size (int): Tamaño del torneo para selección (fijo: 3).
            crossover_prob (float): Probabilidad de cruce (fijo: 0.8).
            random_state (int): Semilla para reproducibilidad.
        """
        self.n_total = n_total
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.random_state = random_state
        
        # REGLA CRÍTICA: Probabilidad de mutación por gen calculada dinámicamente con N_total
        self.mutation_prob = 1.0 / (2.5 * float(self.n_total))
        
        # Generador de números aleatorios para reproducibilidad estricta
        self.rng = np.random.default_rng(seed=self.random_state)

    def initialize_population(self) -> np.ndarray:
        """
        Genera la población inicial de 50 individuos de forma aleatoria.
        Garantiza que ningún individuo tenga 0 características seleccionadas.
        
        Retorna:
            np.ndarray de dimensiones (pop_size, n_total) con valores 0 o 1.
        """
        # Inicialización binaria uniforme (0 o 1 con p=0.5)
        population = self.rng.integers(0, 2, size=(self.pop_size, self.n_total), dtype=np.int8)
        
        # Garantizar que cada individuo tenga al menos una característica seleccionada
        for i in range(self.pop_size):
            if np.sum(population[i]) == 0:
                random_gene = self.rng.integers(0, self.n_total)
                population[i, random_gene] = 1
                
        return population

    def tournament_selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """
        Selecciona un individuo mediante Tournament Selection de tamaño k=3.
        """
        selected_indices = self.rng.choice(self.pop_size, size=self.tournament_size, replace=True)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return np.copy(population[best_index])

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica Single-point crossover con probabilidad crossover_prob (0.8).
        """
        if self.rng.random() < self.crossover_prob:
            # Seleccionar un punto de corte válido entre 1 y n_total - 1
            cut_point = self.rng.integers(1, self.n_total)
            child1 = np.concatenate([parent1[:cut_point], parent2[cut_point:]])
            child2 = np.concatenate([parent2[:cut_point], parent1[cut_point:]])
            return child1, child2
        else:
            return np.copy(parent1), np.copy(parent2)

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Aplica Bit-flip mutation por gen con probabilidad Pm = 1 / (2.5 * N_total).
        Garantiza que después de mutar el cromosoma tenga al menos una característica seleccionada.
        """
        mutated = np.copy(chromosome)
        
        # Generar máscara booleana de mutaciones independientes
        mutation_mask = self.rng.random(size=self.n_total) < self.mutation_prob
        
        # Aplicar bit-flip (1 -> 0, 0 -> 1) donde la máscara es True
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        
        # Garantizar al menos 1 característica seleccionada
        if np.sum(mutated) == 0:
            forced_gene = self.rng.integers(0, self.n_total)
            mutated[forced_gene] = 1
            
        return mutated

    def run(self, evaluator: FitnessEvaluator, experiment_name: str = "AG") -> Dict[str, Any]:
        """
        Ejecuta el Algoritmo Genético.
        
        Parámetros:
            evaluator (FitnessEvaluator): Evaluador con la caché y el clasificador asignado.
            experiment_name (str): Nombre del experimento ('AG-DT' o 'AG-RF').
            
        Retorna:
            Dict[str, Any]: Diccionario con los resultados del AG, mejor cromosoma, historial y estadísticas.
        """
        print("\n" + "=" * 60)
        print(f"INICIO DEL ALGORITMO GENÉTICO: {experiment_name}")
        print(f"Parámetros: Población={self.pop_size}, MaxGen={self.max_generations}, "
              f"Estancamiento={self.stagnation_limit}, Torneo={self.tournament_size}, "
              f"Pc={self.crossover_prob}, Pm={self.mutation_prob:.7f} (1 / 2.5*{self.n_total})")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # 1. Inicializar población
        population = self.initialize_population()
        
        history: List[Dict[str, Any]] = []
        
        best_fitness_overall = -1.0
        best_chromosome_overall = None
        best_recall_overall = 0.0
        best_reduction_overall = 0.0
        best_num_features_overall = 0
        best_generation_found = 0
        
        stagnant_generations = 0
        
        # Bucle generacional (Generaciones 1 a max_generations)
        for gen in range(1, self.max_generations + 1):
            gen_start_time = time.perf_counter()
            
            # Evaluar aptitud de todos los individuos de la población actual
            fitness_scores = np.zeros(self.pop_size, dtype=float)
            recalls = np.zeros(self.pop_size, dtype=float)
            reductions = np.zeros(self.pop_size, dtype=float)
            num_features = np.zeros(self.pop_size, dtype=int)
            
            for i in range(self.pop_size):
                fit, rec, red, n_feats = evaluator.evaluate(population[i])
                fitness_scores[i] = fit
                recalls[i] = rec
                reductions[i] = red
                num_features[i] = n_feats
                
            # Estadísticas de la generación
            best_idx = int(np.argmax(fitness_scores))
            worst_idx = int(np.argmin(fitness_scores))
            
            gen_best_fitness = fitness_scores[best_idx]
            gen_avg_fitness = float(np.mean(fitness_scores))
            gen_worst_fitness = fitness_scores[worst_idx]
            gen_best_recall = recalls[best_idx]
            gen_best_reduction = reductions[best_idx]
            gen_best_n_feats = num_features[best_idx]
            
            # Registrar historial de la generación (Requisito Sección 25)
            log_entry = {
                "generation": gen,
                "best_fitness": gen_best_fitness,
                "average_fitness": gen_avg_fitness,
                "worst_fitness": gen_worst_fitness,
                "best_recall": gen_best_recall,
                "best_num_features": gen_best_n_feats,
                "best_reduction": gen_best_reduction
            }
            history.append(log_entry)
            
            gen_duration = time.perf_counter() - gen_start_time
            print(
                f"[{experiment_name}] Gen {gen:02d}/{self.max_generations} | "
                f"Best Fit: {gen_best_fitness:.5f} | Avg Fit: {gen_avg_fitness:.5f} | "
                f"Worst Fit: {gen_worst_fitness:.5f} | Best Rec: {gen_best_recall:.4f} | "
                f"Feats: {gen_best_n_feats}/{self.n_total} ({gen_best_reduction*100:.1f}% red) | "
                f"Time: {gen_duration:.1f}s"
            )
            
            # Comprobar mejora del mejor fitness histórico
            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_chromosome_overall = np.copy(population[best_idx])
                best_recall_overall = gen_best_recall
                best_reduction_overall = gen_best_reduction
                best_num_features_overall = gen_best_n_feats
                best_generation_found = gen
                stagnant_generations = 0
            else:
                stagnant_generations += 1
                
            # Criterio de parada por estancamiento: 10 generaciones consecutivas sin mejora
            if stagnant_generations >= self.stagnation_limit:
                print(
                    f"\n[{experiment_name}] CRITERIO DE PARADA ALCANZADO: "
                    f"Estancamiento durante {self.stagnation_limit} generaciones consecutivas (Gen {gen})."
                )
                break
                
            # Si alcanzamos la generación máxima, no generar nueva población
            if gen == self.max_generations:
                print(f"\n[{experiment_name}] CRITERIO DE PARADA ALCANZADO: Máximo de {self.max_generations} generaciones.")
                break
                
            # --- CREACIÓN DE LA SIGUIENTE GENERACIÓN ---
            # 1. Elitismo: Guardar el mejor individuo actual para transferirlo intacto
            elite_individual = np.copy(population[best_idx])
            
            new_population = [elite_individual]
            
            # 2. Generar los restantes 49 individuos mediante Selección, Cruce y Mutación
            offspring_needed = self.pop_size - 1
            while len(new_population) < self.pop_size:
                # Selección de 2 progenitores mediante torneo
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Cruce de un solo punto
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutación bit-flip independiente
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
                    
            # Reemplazo generacional
            population = np.array(new_population, dtype=np.int8)
            
        total_ag_time = time.perf_counter() - start_time
        history_df = pd.DataFrame(history)
        cache_stats = evaluator.get_cache_stats()
        
        print("-" * 60)
        print(f"FINALIZÓ {experiment_name}:")
        print(f" - Tiempo total de ejecución del AG: {total_ag_time:.2f} segundos ({total_ag_time/60:.2f} min)")
        print(f" - Generación final alcanzada: {history_df['generation'].iloc[-1]}")
        print(f" - Mejor generación encontrada: Gen {best_generation_found}")
        print(f" - Mejor Fitness: {best_fitness_overall:.5f}")
        print(f" - Características seleccionadas: {best_num_features_overall} de {self.n_total}")
        print(f" - Porcentaje de reducción: {best_reduction_overall*100:.2f}%")
        print(f" - Recall GR=1 (Validation): {best_recall_overall:.4f}")
        print(f" - Estadísticas de Caché: {cache_stats['evaluations_requested']} solicitadas, "
              f"{cache_stats['evaluations_computed']} calculadas, "
              f"{cache_stats['cache_hits']} recuperadas de caché "
              f"({cache_stats['cache_hit_rate_pct']:.1f}% ahorro)")
        print("=" * 60 + "\n")
        
        return {
            "experiment_name": experiment_name,
            "best_chromosome": best_chromosome_overall,
            "best_fitness": best_fitness_overall,
            "best_recall": best_recall_overall,
            "best_reduction": best_reduction_overall,
            "best_num_features": best_num_features_overall,
            "final_generation": int(history_df["generation"].iloc[-1]),
            "best_generation_found": best_generation_found,
            "history_df": history_df,
            "cache_stats": cache_stats,
            "total_time_seconds": total_ag_time
        }
