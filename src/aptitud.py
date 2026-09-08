from typing import Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from src.modelos import ClassifierType


class EvaluadorAptitud:
    
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
        
        self.model_factory = model_factory
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_total = n_total
        self.model_name = model_name
        self.cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.evaluations_requested: int = 0
        self.evaluations_computed: int = 0
        self.cache_hits: int = 0

    def evaluar(self, chromosome: np.ndarray) -> Tuple[float, float, float, int]:
        
        self.evaluations_requested += 1
        chrom_key = tuple(int(gene) for gene in chromosome)
        if chrom_key in self.cache:
            self.cache_hits += 1
            cached_data = self.cache[chrom_key]
            return (
                cached_data["fitness"],
                cached_data["recall"],
                cached_data["reduction"],
                cached_data["n_selected"]
            )
        self.evaluations_computed += 1
        selected_indices = [idx for idx, gene in enumerate(chrom_key) if gene == 1]
        n_selected = len(selected_indices)
        if n_selected == 0:
            return (0.0, 0.0, 0.0, 0)
        X_train_sub = self.X_train.iloc[:, selected_indices]
        X_val_sub = self.X_val.iloc[:, selected_indices]
        model = self.model_factory()
        model.fit(X_train_sub, self.y_train)
        y_val_pred = model.predict(X_val_sub)
        recall = float(recall_score(self.y_val, y_val_pred, pos_label=1, zero_division=0))
        reduction = float(1.0 - (n_selected / self.n_total))
        fitness = float(0.7 * recall + 0.3 * reduction)
        self.cache[chrom_key] = {
            "fitness": fitness,
            "recall": recall,
            "reduction": reduction,
            "n_selected": n_selected
        }
        
        return (fitness, recall, reduction, n_selected)

    def obtener_estadisticas_cache(self) -> Dict[str, Any]:
        
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
