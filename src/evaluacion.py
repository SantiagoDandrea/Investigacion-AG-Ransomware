import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.modelos import obtener_arbol_decision, obtener_bosque_aleatorio, obtener_xgboost, ClassifierType


def evaluar_modelo_individual(
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
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start
    pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - pred_start
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    reduction_pct = (1.0 - (n_selected / n_total)) * 100.0
    
    return {
        "conjunto_caracteristicas": feature_set_name,
        "modelo": model_name,
        "numero_caracteristicas": n_selected,
        "reduccion_porcentaje": reduction_pct,
        "exactitud": acc,
        "precision": prec,
        "recall": rec,
        "puntuacion_f1": f1,
        "tiempo_entrenamiento_seg": training_time,
        "tiempo_prediccion_seg": prediction_time,
        "aptitud_validacion": fitness_val if fitness_val is not None else np.nan
    }


def ejecutar_evaluacion_referencia(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_total: int,
    random_state: int = 42
) -> pd.DataFrame:
    
    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL BASELINE (TODAS LAS CARACTERÍSTICAS)")
    print(f"Características utilizadas: {n_total} (después de eliminar constantes)")
    print("=" * 60)
    
    models = [
        ("Arbol de decision", obtener_arbol_decision(random_state=random_state)),
        ("Random Forest", obtener_bosque_aleatorio(random_state=random_state)),
        ("XGBoost", obtener_xgboost(random_state=random_state))
    ]
    
    baseline_records = []
    for model_name, model in models:
        res = evaluar_modelo_individual(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            feature_set_name="Todas las características",
            n_selected=n_total,
            n_total=n_total,
            fitness_val=None
        )
        baseline_records.append(res)
        print(
            f"Referencia | {model_name:15s} | Exactitud: {res['exactitud']:.4f} | "
            f"Precisión: {res['precision']:.4f} | Recall: {res['recall']:.4f} | "
            f"F1: {res['puntuacion_f1']:.4f} | Tiempo de entrenamiento: {res['tiempo_entrenamiento_seg']:.4f}s | "
            f"Tiempo de predicción: {res['tiempo_prediccion_seg']:.5f}s"
        )
        
    df_baseline = pd.DataFrame(baseline_records)
    return df_baseline


def ejecutar_matriz_evaluacion_final(
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
    
    print("\n" + "=" * 80)
    print("EVALUACIÓN FINAL CRUZADA SOBRE EL CONJUNTO DE PRUEBA")
    print("=" * 80)
    all_indices = list(range(n_total))
    dt_indices = [idx for idx, gene in enumerate(best_chrom_dt) if gene == 1]
    rf_indices = [idx for idx, gene in enumerate(best_chrom_rf) if gene == 1]
    
    feature_sets = [
        ("Todas las características", all_indices, n_total, None),
        ("Subconjunto AG-DT", dt_indices, len(dt_indices), fitness_dt),
        ("Subconjunto AG-RF", rf_indices, len(rf_indices), fitness_rf)
    ]
    
    final_records = []
    
    for set_name, indices, n_sel, fit_val in feature_sets:
        print(f"\n--- Evaluando {set_name} ({n_sel} características, {(1 - n_sel/n_total)*100:.2f}% reducción) ---")
        
        X_tr_sub = X_train.iloc[:, indices]
        X_te_sub = X_test.iloc[:, indices]
        models = [
            ("Arbol de decision", obtener_arbol_decision(random_state=random_state)),
            ("Random Forest", obtener_bosque_aleatorio(random_state=random_state)),
            ("XGBoost", obtener_xgboost(random_state=random_state))
        ]
        
        for model_name, model in models:
            res = evaluar_modelo_individual(
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
                f"{set_name:25s} | {model_name:15s} | Exactitud: {res['exactitud']:.4f} | "
                f"Precisión: {res['precision']:.4f} | Recall: {res['recall']:.4f} | "
                f"F1: {res['puntuacion_f1']:.4f} | Tiempo de entrenamiento: {res['tiempo_entrenamiento_seg']:.4f}s | "
                f"Tiempo de predicción: {res['tiempo_prediccion_seg']:.5f}s"
            )
            
    df_final = pd.DataFrame(final_records)
    return df_final
