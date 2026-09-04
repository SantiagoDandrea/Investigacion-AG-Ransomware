"""
Módulo fitness.py
Responsabilidad: Evaluación del fitness de individuos, cálculo de Recall (GR=1),
Reduction y gestión de la caché de evaluaciones.
"""

from typing import Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from src.models import ClassifierType


class FitnessEvaluator:
    """
    Evaluador de fitness con caché de evaluaciones para el Algoritmo Genético.
    
    Cada ejecución del AG (AG-DT o AG-RF) debe tener su propia instancia independiente
    de FitnessEvaluator para no compartir resultados de clasificadores distintos.
    """
    def __init__(
        self,
        model_factory: Callable[[], ClassifierType],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_total: int,
        model_name: str = "Classifier"
    ):
        """
        Parámetros:
            model_factory: Función que instancia un nuevo modelo (DT o RF) con hiperparámetros fijos.
            X_train: DataFrame de entrenamiento (con las N_total columnas activas).
            y_train: Serie con target de entrenamiento.
            X_val: DataFrame de validación (con las N_total columnas activas).
            y_val: Serie con target de validación.
            n_total: Cantidad de características tras el preprocesamiento (dinámico).
            model_name: Nombre descriptivo del clasificador (ej. 'Decision Tree' o 'Random Forest').
        """
        self.model_factory = model_factory
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_total = n_total
        self.model_name = model_name
        
        # Caché de evaluaciones: clave = tuple(chromosome), valor = dict con métricas
        self.cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        
        # Métricas de uso de la caché
        self.evaluations_requested: int = 0
        self.evaluations_computed: int = 0
        self.cache_hits: int = 0

    def evaluate(self, chromosome: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Evalúa un individuo representado por un cromosoma binario.
        
        Fórmula:
            Fitness = 0.7 * Recall (GR=1) + 0.3 * Reduction
            Reduction = 1 - (N_selected / N_total)
            
        Retorna:
            Tuple[float, float, float, int]:
                - fitness: Valor de aptitud [0.0, 1.0].
                - recall: Recall de la clase positiva GR=1 en Validation.
                - reduction: Proporción de reducción de características.
                - n_selected: Cantidad de características seleccionadas.
        """
        self.evaluations_requested += 1
        
        # Convertir a tupla hashable para indexar en la caché
        chrom_key = tuple(int(gene) for gene in chromosome)
        
        # 1. Comprobar si ya está en caché
        if chrom_key in self.cache:
            self.cache_hits += 1
            cached_data = self.cache[chrom_key]
            return (
                cached_data["fitness"],
                cached_data["recall"],
                cached_data["reduction"],
                cached_data["n_selected"]
            )
            
        # 2. Si no está en caché, calcular
        self.evaluations_computed += 1
        
        # Identificar índices de genes activos (valor == 1)
        selected_indices = [idx for idx, gene in enumerate(chrom_key) if gene == 1]
        n_selected = len(selected_indices)
        
        # Salvaguarda metodológica: no permitir cromosomas vacíos
        if n_selected == 0:
            # Penalización extrema si ocurre por algún motivo anómalo
            return (0.0, 0.0, 0.0, 0)
            
        # Subconjunto de características para Train y Validation
        # Usamos iloc para indexación posicional rápida
        X_train_sub = self.X_train.iloc[:, selected_indices]
        X_val_sub = self.X_val.iloc[:, selected_indices]
        
        # Instanciar y entrenar el clasificador correspondiente
        model = self.model_factory()
        model.fit(X_train_sub, self.y_train)
        
        # Predecir sobre Validation
        y_val_pred = model.predict(X_val_sub)
        
        # Calcular Recall de la clase positiva: GR = 1 (Ransomware)
        # REGLA: No usar macro ni weighted, solo clase positiva pos_label=1
        recall = float(recall_score(self.y_val, y_val_pred, pos_label=1, zero_division=0))
        
        # Calcular Reducción
        reduction = float(1.0 - (n_selected / self.n_total))
        
        # Calcular Fitness ponderado
        fitness = float(0.7 * recall + 0.3 * reduction)
        
        # Guardar en caché
        self.cache[chrom_key] = {
            "fitness": fitness,
            "recall": recall,
            "reduction": reduction,
            "n_selected": n_selected
        }
        
        return (fitness, recall, reduction, n_selected)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retorna las estadísticas del uso de la caché para este evaluador.
        """
        hit_rate = (
            (self.cache_hits / self.evaluations_requested * 100)
            if self.evaluations_requested > 0 else 0.0
        )
        return {
            "model_name": self.model_name,
            "evaluations_requested": self.evaluations_requested,
            "evaluations_computed": self.evaluations_computed,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_pct": hit_rate,
            "unique_chromosomes_cached": len(self.cache)
        }
