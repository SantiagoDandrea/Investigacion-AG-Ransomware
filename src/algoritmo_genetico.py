import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.aptitud import EvaluadorAptitud


class AlgoritmoGenetico:
    
    def __init__(
        self,
        total_caracteristicas: int,
        tamano_poblacion: int = 50,
        max_generaciones: int = 50,
        limite_estancamiento: int = 10,
        tamano_torneo: int = 3,
        probabilidad_cruce: float = 0.8,
        semilla_aleatoria: int = 42
    ):
        
        self.total_caracteristicas = total_caracteristicas
        self.tamano_poblacion = tamano_poblacion
        self.max_generaciones = max_generaciones
        self.limite_estancamiento = limite_estancamiento
        self.tamano_torneo = tamano_torneo
        self.probabilidad_cruce = probabilidad_cruce
        self.semilla_aleatoria = semilla_aleatoria
        self.probabilidad_mutacion = 1.0 / (2.5 * float(self.total_caracteristicas))
        self.generador_aleatorio = np.random.default_rng(seed=self.semilla_aleatoria)

    def inicializar_poblacion(self) -> np.ndarray:
        poblacion = self.generador_aleatorio.integers(0, 2, size=(self.tamano_poblacion, self.total_caracteristicas), dtype=np.int8)
        for indice in range(self.tamano_poblacion):
            if np.sum(poblacion[indice]) == 0:
                gen_aleatorio = self.generador_aleatorio.integers(0, self.total_caracteristicas)
                poblacion[indice, gen_aleatorio] = 1
                
        return poblacion

    def seleccion_torneo(self, poblacion: np.ndarray, puntuaciones_aptitud: np.ndarray) -> np.ndarray:
        
        indices_seleccionados = self.generador_aleatorio.choice(self.tamano_poblacion, size=self.tamano_torneo, replace=True)
        mejor_indice = indices_seleccionados[np.argmax(puntuaciones_aptitud[indices_seleccionados])]
        return np.copy(poblacion[mejor_indice])

    def cruzar(self, progenitor_1: np.ndarray, progenitor_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.generador_aleatorio.random() < self.probabilidad_cruce:
            punto_corte = self.generador_aleatorio.integers(1, self.total_caracteristicas)
            descendiente_1 = np.concatenate([progenitor_1[:punto_corte], progenitor_2[punto_corte:]])
            descendiente_2 = np.concatenate([progenitor_2[:punto_corte], progenitor_1[punto_corte:]])
            return descendiente_1, descendiente_2
        else:
            return np.copy(progenitor_1), np.copy(progenitor_2)

    def mutar(self, cromosoma: np.ndarray) -> np.ndarray:
        
        mutado = np.copy(cromosoma)
        mascara_mutacion = self.generador_aleatorio.random(size=self.total_caracteristicas) < self.probabilidad_mutacion
        mutado[mascara_mutacion] = 1 - mutado[mascara_mutacion]
        if np.sum(mutado) == 0:
            gen_forzado = self.generador_aleatorio.integers(0, self.total_caracteristicas)
            mutado[gen_forzado] = 1
            
        return mutado

    def ejecutar(self, evaluador: EvaluadorAptitud, nombre_experimento: str = "AG") -> Dict[str, Any]:
        
        print("\n" + "=" * 60)
        print(f"INICIO DEL ALGORITMO GENÉTICO: {nombre_experimento}")
        print(f"Parámetros: Población={self.tamano_poblacion}, MaxGen={self.max_generaciones}, Estancamiento={self.limite_estancamiento}, Torneo={self.tamano_torneo}, Pc={self.probabilidad_cruce}, Pm={self.probabilidad_mutacion:.7f} (1 / 2.5*{self.total_caracteristicas})")
        print("=" * 60)
        
        tiempo_inicio = time.perf_counter()
        poblacion = self.inicializar_poblacion()
        
        historial: List[Dict[str, Any]] = []
        
        mejor_aptitud_global = -1.0
        mejor_cromosoma_global = None
        mejor_recall_global = 0.0
        mejor_reduccion_global = 0.0
        mejor_num_caracteristicas_global = 0
        mejor_generacion_encontrada = 0
        
        generaciones_estancadas = 0
        for generacion in range(1, self.max_generaciones + 1):
            inicio_generacion = time.perf_counter()
            puntuaciones_aptitud = np.zeros(self.tamano_poblacion, dtype=float)
            recuperaciones = np.zeros(self.tamano_poblacion, dtype=float)
            reducciones = np.zeros(self.tamano_poblacion, dtype=float)
            cantidad_caracteristicas = np.zeros(self.tamano_poblacion, dtype=int)
            
            for indice in range(self.tamano_poblacion):
                aptitud, recall, reduccion, cantidad = evaluador.evaluar(poblacion[indice])
                puntuaciones_aptitud[indice] = aptitud
                recuperaciones[indice] = recall
                reducciones[indice] = reduccion
                cantidad_caracteristicas[indice] = cantidad
            mejor_indice = int(np.argmax(puntuaciones_aptitud))
            peor_indice = int(np.argmin(puntuaciones_aptitud))
            
            mejor_aptitud_generacion = puntuaciones_aptitud[mejor_indice]
            aptitud_promedio_generacion = float(np.mean(puntuaciones_aptitud))
            peor_aptitud_generacion = puntuaciones_aptitud[peor_indice]
            mejor_recall_generacion = recuperaciones[mejor_indice]
            mejor_reduccion_generacion = reducciones[mejor_indice]
            mejor_cantidad_generacion = cantidad_caracteristicas[mejor_indice]
            entrada_historial = {
                "generacion": generacion,
                "mejor_aptitud": mejor_aptitud_generacion,
                "aptitud_promedio": aptitud_promedio_generacion,
                "peor_aptitud": peor_aptitud_generacion,
                "mejor_recall": mejor_recall_generacion,
                "mejor_num_caracteristicas": mejor_cantidad_generacion,
                "mejor_reduccion": mejor_reduccion_generacion
            }
            historial.append(entrada_historial)
            
            duracion_generacion = time.perf_counter() - inicio_generacion
            print(
                f"[{nombre_experimento}] Generación {generacion:02d}/{self.max_generaciones} | "
                f"Mejor aptitud: {mejor_aptitud_generacion:.5f} | Aptitud promedio: {aptitud_promedio_generacion:.5f} | "
                f"Peor aptitud: {peor_aptitud_generacion:.5f} | Mejor recuperación: {mejor_recall_generacion:.4f} | "
                f"Características: {mejor_cantidad_generacion}/{self.total_caracteristicas} ({mejor_reduccion_generacion*100:.1f}% reducción) | "
                f"Tiempo: {duracion_generacion:.1f}s"
            )
            if mejor_aptitud_generacion > mejor_aptitud_global:
                mejor_aptitud_global = mejor_aptitud_generacion
                mejor_cromosoma_global = np.copy(poblacion[mejor_indice])
                mejor_recall_global = mejor_recall_generacion
                mejor_reduccion_global = mejor_reduccion_generacion
                mejor_num_caracteristicas_global = mejor_cantidad_generacion
                mejor_generacion_encontrada = generacion
                generaciones_estancadas = 0
            else:
                generaciones_estancadas += 1
            if generaciones_estancadas >= self.limite_estancamiento:
                print(
                    f"\n[{nombre_experimento}] CRITERIO DE PARADA ALCANZADO: "
                    f"Estancamiento durante {self.limite_estancamiento} generaciones consecutivas (Generación {generacion})."
                )
                break
            if generacion == self.max_generaciones:
                print(f"\n[{nombre_experimento}] CRITERIO DE PARADA ALCANZADO: Máximo de {self.max_generaciones} generaciones.")
                break
            individuo_destacado = np.copy(poblacion[mejor_indice])
            
            nueva_poblacion = [individuo_destacado]
            while len(nueva_poblacion) < self.tamano_poblacion:
                progenitor_1 = self.seleccion_torneo(poblacion, puntuaciones_aptitud)
                progenitor_2 = self.seleccion_torneo(poblacion, puntuaciones_aptitud)
                descendiente_1, descendiente_2 = self.cruzar(progenitor_1, progenitor_2)
                descendiente_1 = self.mutar(descendiente_1)
                descendiente_2 = self.mutar(descendiente_2)
                
                nueva_poblacion.append(descendiente_1)
                if len(nueva_poblacion) < self.tamano_poblacion:
                    nueva_poblacion.append(descendiente_2)
            poblacion = np.array(nueva_poblacion, dtype=np.int8)
            
        tiempo_total = time.perf_counter() - tiempo_inicio
        historial_df = pd.DataFrame(historial)
        estadisticas_cache = evaluador.obtener_estadisticas_cache()
        
        print("-" * 60)
        print(f"FINALIZÓ {nombre_experimento}:")
        print(f" - Tiempo total de ejecución del AG: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} min)")
        print(f" - Generación final alcanzada: {historial_df['generacion'].iloc[-1]}")
        print(f" - Mejor generación encontrada: {mejor_generacion_encontrada}")
        print(f" - Mejor aptitud: {mejor_aptitud_global:.5f}")
        print(f" - Características seleccionadas: {mejor_num_caracteristicas_global} de {self.total_caracteristicas}")
        print(f" - Porcentaje de reducción: {mejor_reduccion_global*100:.2f}%")
        print(f" - Recuperación GR=1 (Validación): {mejor_recall_global:.4f}")
        print(f" - Estadísticas de la caché: {estadisticas_cache['evaluations_requested']} solicitadas, "
            f"{estadisticas_cache['evaluations_computed']} calculadas, "
            f"{estadisticas_cache['cache_hits']} recuperadas de caché "
            f"({estadisticas_cache['cache_hit_rate_pct']:.1f}% ahorro)")
        print("=" * 60 + "\n")
        
        return {
            "nombre_experimento": nombre_experimento,
            "mejor_cromosoma": mejor_cromosoma_global,
            "mejor_aptitud": mejor_aptitud_global,
            "mejor_recall": mejor_recall_global,
            "mejor_reduccion": mejor_reduccion_global,
            "mejor_num_caracteristicas": mejor_num_caracteristicas_global,
            "generacion_final": int(historial_df["generacion"].iloc[-1]),
            "mejor_generacion_encontrada": mejor_generacion_encontrada,
            "historial": historial_df,
            "estadisticas_cache": estadisticas_cache,
            "tiempo_total_segundos": tiempo_total
        }
