"""
Módulo evaluation.py
Responsabilidad: Evaluación del baseline y evaluación final de combinaciones
sobre el conjunto Test aislado, con medición de métricas de rendimiento y tiempos.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models import get_decision_tree, get_random_forest, get_xgboost, ClassifierType


def evaluate_single_model(
    model: ClassifierType,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_set_name: str,
    n_selected: int,
    n_total: int,
    fitness_val: float = None
) -> Dict[str, Any]:
    """
    Entrena un modelo, realiza predicciones sobre Test, calcula métricas de clasificación
    y mide los tiempos de entrenamiento y predicción.
    """
    # 1. Medir tiempo de entrenamiento
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start
    
    # 2. Medir tiempo de predicción sobre el conjunto Test
    pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - pred_start
    
    # 3. Calcular métricas de rendimiento
    acc = accuracy_score(y_test, y_pred)
    # REGLA CRÍTICA: Precision, Recall y F1 específicos de la clase positiva GR=1 (Ransomware)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Reducción de características
    reduction_pct = (1.0 - (n_selected / n_total)) * 100.0
    
    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "num_features": n_selected,
        "reduction_pct": reduction_pct,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "training_time_sec": training_time,
        "prediction_time_sec": prediction_time,
        "fitness_validation": fitness_val if fitness_val is not None else np.nan
    }


def run_baseline_evaluation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_total: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evalúa el baseline utilizando todas las N_total características sobre el conjunto Test.
    Modelos evaluados: Decision Tree, Random Forest y XGBoost.
    """
    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL BASELINE (TODAS LAS CARACTERÍSTICAS)")
    print(f"Características utilizadas: {n_total} (después de eliminar constantes)")
    print("=" * 60)
    
    models = [
        ("Decision Tree", get_decision_tree(random_state=random_state)),
        ("Random Forest", get_random_forest(random_state=random_state)),
        ("XGBoost", get_xgboost(random_state=random_state))
    ]
    
    baseline_records = []
    for model_name, model in models:
        res = evaluate_single_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            feature_set_name="Todas las features",
            n_selected=n_total,
            n_total=n_total,
            fitness_val=None
        )
        baseline_records.append(res)
        print(
            f"Baseline | {model_name:15s} | Acc: {res['accuracy']:.4f} | "
            f"Prec: {res['precision']:.4f} | Recall: {res['recall']:.4f} | "
            f"F1: {res['f1']:.4f} | Train Time: {res['training_time_sec']:.4f}s | "
            f"Pred Time: {res['prediction_time_sec']:.5f}s"
        )
        
    df_baseline = pd.DataFrame(baseline_records)
    return df_baseline


def run_final_evaluation_matrix(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_total: int,
    best_chrom_dt: np.ndarray,
    best_chrom_rf: np.ndarray,
    fitness_dt: float,
    fitness_rf: float,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Ejecuta la evaluación final cruzada sobre el conjunto Test para:
    1. Todas las features (Baseline)
    2. Subconjunto A (Optimizado con AG-DT)
    3. Subconjunto B (Optimizado con AG-RF)
    
    Cada subconjunto se evalúa con:
    - Decision Tree
    - Random Forest
    - XGBoost
    
    Total de combinaciones evaluadas: 3 subconjuntos x 3 modelos = 9 evaluaciones.
    """
    print("\n" + "=" * 80)
    print("EVALUACIÓN FINAL CRUZADA SOBRE EL CONJUNTO TEST")
    print("=" * 80)
    
    # Subconjuntos e índices de características activas
    all_indices = list(range(n_total))
    dt_indices = [idx for idx, gene in enumerate(best_chrom_dt) if gene == 1]
    rf_indices = [idx for idx, gene in enumerate(best_chrom_rf) if gene == 1]
    
    feature_sets = [
        ("Todas las features", all_indices, n_total, None),
        ("Subconjunto AG-DT", dt_indices, len(dt_indices), fitness_dt),
        ("Subconjunto AG-RF", rf_indices, len(rf_indices), fitness_rf)
    ]
    
    final_records = []
    
    for set_name, indices, n_sel, fit_val in feature_sets:
        print(f"\n--- Evaluando {set_name} ({n_sel} features, {(1 - n_sel/n_total)*100:.2f}% reducción) ---")
        
        X_tr_sub = X_train.iloc[:, indices]
        X_te_sub = X_test.iloc[:, indices]
        
        # Modelos frescos con los hiperparámetros fijos
        models = [
            ("Decision Tree", get_decision_tree(random_state=random_state)),
            ("Random Forest", get_random_forest(random_state=random_state)),
            ("XGBoost", get_xgboost(random_state=random_state))
        ]
        
        for model_name, model in models:
            res = evaluate_single_model(
                model=model,
                X_train=X_tr_sub,
                y_train=y_train,
                X_test=X_te_sub,
                y_test=y_test,
                model_name=model_name,
                feature_set_name=set_name,
                n_selected=n_sel,
                n_total=n_total,
                fitness_val=fit_val
            )
            final_records.append(res)
            print(
                f"{set_name:20s} | {model_name:15s} | Acc: {res['accuracy']:.4f} | "
                f"Prec: {res['precision']:.4f} | Recall: {res['recall']:.4f} | "
                f"F1: {res['f1']:.4f} | Train Time: {res['training_time_sec']:.4f}s | "
                f"Pred Time: {res['prediction_time_sec']:.5f}s"
            )
            
    df_final = pd.DataFrame(final_records)
    return df_final
