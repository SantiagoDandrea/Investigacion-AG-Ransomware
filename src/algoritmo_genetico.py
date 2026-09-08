import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.aptitud import EvaluadorAptitud


class AlgoritmoGenetico:
    
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
        
        self.n_total = n_total
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.random_state = random_state
        self.mutation_prob = 1.0 / (2.5 * float(self.n_total))
        self.rng = np.random.default_rng(seed=self.random_state)

    def inicializar_poblacion(self) -> np.ndarray:
        population = self.rng.integers(0, 2, size=(self.pop_size, self.n_total), dtype=np.int8)
        for i in range(self.pop_size):
            if np.sum(population[i]) == 0:
                random_gene = self.rng.integers(0, self.n_total)
                population[i, random_gene] = 1
                
        return population

    def seleccion_torneo(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        
        selected_indices = self.rng.choice(self.pop_size, size=self.tournament_size, replace=True)
        best_index = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return np.copy(population[best_index])

    def cruzar(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.rng.random() < self.crossover_prob:
            cut_point = self.rng.integers(1, self.n_total)
            child1 = np.concatenate([parent1[:cut_point], parent2[cut_point:]])
            child2 = np.concatenate([parent2[:cut_point], parent1[cut_point:]])
            return child1, child2
        else:
            return np.copy(parent1), np.copy(parent2)

    def mutar(self, chromosome: np.ndarray) -> np.ndarray:
        
        mutated = np.copy(chromosome)
        mutation_mask = self.rng.random(size=self.n_total) < self.mutation_prob
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        if np.sum(mutated) == 0:
            forced_gene = self.rng.integers(0, self.n_total)
            mutated[forced_gene] = 1
            
        return mutated

    def ejecutar(self, evaluator: EvaluadorAptitud, experiment_name: str = "AG") -> Dict[str, Any]:
        
        print("\n" + "=" * 60)
        print(f"INICIO DEL ALGORITMO GENÉTICO: {experiment_name}")
        print(f"Parámetros: Población={self.pop_size}, MaxGen={self.max_generations}, "
              f"Estancamiento={self.stagnation_limit}, Torneo={self.tournament_size}, "
              f"Pc={self.crossover_prob}, Pm={self.mutation_prob:.7f} (1 / 2.5*{self.n_total})")
        print("=" * 60)
        
        start_time = time.perf_counter()
        population = self.inicializar_poblacion()
        
        history: List[Dict[str, Any]] = []
        
        best_fitness_overall = -1.0
        best_chromosome_overall = None
        best_recall_overall = 0.0
        best_reduction_overall = 0.0
        best_num_features_overall = 0
        best_generation_found = 0
        
        stagnant_generations = 0
        for gen in range(1, self.max_generations + 1):
            gen_start_time = time.perf_counter()
            fitness_scores = np.zeros(self.pop_size, dtype=float)
            recalls = np.zeros(self.pop_size, dtype=float)
            reductions = np.zeros(self.pop_size, dtype=float)
            num_features = np.zeros(self.pop_size, dtype=int)
            
            for i in range(self.pop_size):
                fit, rec, red, n_feats = evaluator.evaluar(population[i])
                fitness_scores[i] = fit
                recalls[i] = rec
                reductions[i] = red
                num_features[i] = n_feats
            best_idx = int(np.argmax(fitness_scores))
            worst_idx = int(np.argmin(fitness_scores))
            
            gen_best_fitness = fitness_scores[best_idx]
            gen_avg_fitness = float(np.mean(fitness_scores))
            gen_worst_fitness = fitness_scores[worst_idx]
            gen_best_recall = recalls[best_idx]
            gen_best_reduction = reductions[best_idx]
            gen_best_n_feats = num_features[best_idx]
            log_entry = {
                "generacion": gen,
                "mejor_aptitud": gen_best_fitness,
                "aptitud_promedio": gen_avg_fitness,
                "peor_aptitud": gen_worst_fitness,
                "mejor_recall": gen_best_recall,
                "mejor_num_caracteristicas": gen_best_n_feats,
                "mejor_reduccion": gen_best_reduction
            }
            history.append(log_entry)
            
            gen_duration = time.perf_counter() - gen_start_time
            print(
                f"[{experiment_name}] Gen {gen:02d}/{self.max_generations} | "
                f"Mejor aptitud: {gen_best_fitness:.5f} | Aptitud promedio: {gen_avg_fitness:.5f} | "
                f"Peor aptitud: {gen_worst_fitness:.5f} | Mejor recall: {gen_best_recall:.4f} | "
                f"Características: {gen_best_n_feats}/{self.n_total} ({gen_best_reduction*100:.1f}% reducción) | "
                f"Tiempo: {gen_duration:.1f}s"
            )
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
            if stagnant_generations >= self.stagnation_limit:
                print(
                    f"\n[{experiment_name}] CRITERIO DE PARADA ALCANZADO: "
                    f"Estancamiento durante {self.stagnation_limit} generaciones consecutivas (Gen {gen})."
                )
                break
            if gen == self.max_generations:
                print(f"\n[{experiment_name}] CRITERIO DE PARADA ALCANZADO: Máximo de {self.max_generations} generaciones.")
                break
            elite_individual = np.copy(population[best_idx])
            
            new_population = [elite_individual]
            offspring_needed = self.pop_size - 1
            while len(new_population) < self.pop_size:
                parent1 = self.seleccion_torneo(population, fitness_scores)
                parent2 = self.seleccion_torneo(population, fitness_scores)
                child1, child2 = self.cruzar(parent1, parent2)
                child1 = self.mutar(child1)
                child2 = self.mutar(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            population = np.array(new_population, dtype=np.int8)
            
        total_ag_time = time.perf_counter() - start_time
        history_df = pd.DataFrame(history)
        cache_stats = evaluator.obtener_estadisticas_cache()
        
        print("-" * 60)
        print(f"FINALIZÓ {experiment_name}:")
        print(f" - Tiempo total de ejecución del AG: {total_ag_time:.2f} segundos ({total_ag_time/60:.2f} min)")
        print(f" - Generación final alcanzada: {history_df['generacion'].iloc[-1]}")
        print(f" - Mejor generación encontrada: Gen {best_generation_found}")
        print(f" - Mejor aptitud: {best_fitness_overall:.5f}")
        print(f" - Características seleccionadas: {best_num_features_overall} de {self.n_total}")
        print(f" - Porcentaje de reducción: {best_reduction_overall*100:.2f}%")
        print(f" - Recall GR=1 (Validación): {best_recall_overall:.4f}")
        print(f" - Estadísticas de la caché: {cache_stats['evaluations_requested']} solicitadas, "
              f"{cache_stats['evaluations_computed']} calculadas, "
              f"{cache_stats['cache_hits']} recuperadas de caché "
              f"({cache_stats['cache_hit_rate_pct']:.1f}% ahorro)")
        print("=" * 60 + "\n")
        
        return {
            "nombre_experimento": experiment_name,
            "mejor_cromosoma": best_chromosome_overall,
            "mejor_aptitud": best_fitness_overall,
            "mejor_recall": best_recall_overall,
            "mejor_reduccion": best_reduction_overall,
            "mejor_num_caracteristicas": best_num_features_overall,
            "generacion_final": int(history_df["generacion"].iloc[-1]),
            "mejor_generacion_encontrada": best_generation_found,
            "historial": history_df,
            "estadisticas_cache": cache_stats,
            "tiempo_total_segundos": total_ag_time
        }
