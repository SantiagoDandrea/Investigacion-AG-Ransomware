"""
Módulo main.py
Responsabilidad: Orquestar el experimento completo de investigación:
1. Carga e inspección del dataset PE Headers.
2. Preprocesamiento, exclusión de metadatos, remoción de constantes y split estratificado.
3. Evaluación del Baseline (todas las características) con DT, RF y XGBoost.
4. Optimización mediante Algoritmo Genético con Decision Tree (AG-DT) -> Subconjunto A.
5. Optimización mediante Algoritmo Genético con Random Forest (AG-RF) -> Subconjunto B.
6. Evaluación final cruzada sobre el conjunto Test para todas las combinaciones.
7. Almacenamiento estructurado de resultados en 'results/'.
8. Generación de gráficos de convergencia y reporte académico.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.models import get_decision_tree, get_random_forest, get_xgboost
from src.fitness import FitnessEvaluator
from src.genetic_algorithm import GeneticAlgorithm
from src.evaluation import run_baseline_evaluation, run_final_evaluation_matrix


def save_selected_features(
    feature_names: list,
    chromosome: np.ndarray,
    output_path: str,
    ag_name: str
) -> pd.DataFrame:
    """
    Guarda en un archivo CSV las características seleccionadas (gen == 1).
    """
    selected = [
        {"feature_index_active": idx, "feature_name": feature_names[idx]}
        for idx, gene in enumerate(chromosome) if gene == 1
    ]
    df_sel = pd.DataFrame(selected)
    df_sel.to_csv(output_path, index=False)
    print(f"[{ag_name}] Subconjunto guardado en: {output_path} ({len(df_sel)} características)")
    return df_sel


def plot_convergence(history_dt: pd.DataFrame, history_rf: pd.DataFrame, output_path: str):
    """
    Genera gráficos comparativos de convergencia para AG-DT y AG-RF.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # AG-DT
    axes[0].plot(history_dt["generation"], history_dt["best_fitness"], label="Mejor Fitness", color="#1f77b4", lw=2)
    axes[0].plot(history_dt["generation"], history_dt["average_fitness"], label="Fitness Promedio", color="#aec7e8", linestyle="--")
    axes[0].plot(history_dt["generation"], history_dt["worst_fitness"], label="Peor Fitness", color="#c7c7c7", linestyle=":")
    axes[0].set_title("Evolución AG-DT (Evaluador: Decision Tree)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Generación", fontsize=11)
    axes[0].set_ylabel("Fitness (0.7*Recall + 0.3*Reduction)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")
    
    # AG-RF
    axes[1].plot(history_rf["generation"], history_rf["best_fitness"], label="Mejor Fitness", color="#2ca02c", lw=2)
    axes[1].plot(history_rf["generation"], history_rf["average_fitness"], label="Fitness Promedio", color="#98df8a", linestyle="--")
    axes[1].plot(history_rf["generation"], history_rf["worst_fitness"], label="Peor Fitness", color="#c7c7c7", linestyle=":")
    axes[1].set_title("Evolución AG-RF (Evaluador: Random Forest)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Generación", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[REPORTE] Gráfico de convergencia guardado en: {output_path}")


def main():
    print("=" * 80)
    print("PROYECTO DE INVESTIGACIÓN UNIVERSITARIO")
    print("Optimización de la selección de características para la detección de ransomware")
    print("mediante Algoritmos Genéticos (PE Headers)")
    print("=" * 80)
    
    # Crear directorio results si no existe
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. CARGA DE DATOS (Módulo data_loader)
    data_path = "data/Ransomware_headers.csv"
    df_raw, data_info = load_raw_data(data_path)
    
    # 2. PREPROCESAMIENTO Y SPLIT (Módulo preprocessing)
    # Semilla fija para estricta reproducibilidad
    RANDOM_STATE = 42
    preprocessed = preprocess_data(df_raw, random_state=RANDOM_STATE)
    
    n_total = preprocessed.n_total
    active_features = preprocessed.feature_names
    
    # 3. EVALUACIÓN DEL BASELINE (Módulo evaluation)
    # Se evalúa sobre el conjunto Test aislado utilizando todas las N_total características
    df_baseline = run_baseline_evaluation(
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_test=preprocessed.X_test,
        y_test=preprocessed.y_test,
        n_total=n_total,
        random_state=RANDOM_STATE
    )
    df_baseline.to_csv(os.path.join(results_dir, "baseline_results.csv"), index=False)
    print(f"[BASELINE] Resultados guardados en: {os.path.join(results_dir, 'baseline_results.csv')}")
    
    # 4. PRIMERA EJECUCIÓN DEL ALGORITMO GENÉTICO: AG-DT (Subconjunto A)
    # Evaluador de fitness específico para Decision Tree con su propia caché
    evaluator_dt = FitnessEvaluator(
        model_factory=lambda: get_decision_tree(random_state=RANDOM_STATE),
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_val=preprocessed.X_val,
        y_val=preprocessed.y_val,
        n_total=n_total,
        model_name="Decision Tree"
    )
    
    ag_dt = GeneticAlgorithm(
        n_total=n_total,
        pop_size=50,
        max_generations=50,
        stagnation_limit=10,
        tournament_size=3,
        crossover_prob=0.8,
        random_state=RANDOM_STATE
    )
    
    res_ag_dt = ag_dt.run(evaluator=evaluator_dt, experiment_name="AG-DT")
    
    # Guardar historial y subconjunto A
    res_ag_dt["history_df"].to_csv(os.path.join(results_dir, "ag_dt_history.csv"), index=False)
    save_selected_features(
        feature_names=active_features,
        chromosome=res_ag_dt["best_chromosome"],
        output_path=os.path.join(results_dir, "selected_features_ag_dt.csv"),
        ag_name="AG-DT"
    )
    
    # Guardar métricas resumen de AG-DT
    ag_dt_summary = pd.DataFrame([{
        "experiment": "AG-DT",
        "evaluator_model": "Decision Tree",
        "best_fitness": res_ag_dt["best_fitness"],
        "best_recall_validation": res_ag_dt["best_recall"],
        "num_features_selected": res_ag_dt["best_num_features"],
        "total_features": n_total,
        "reduction_pct": res_ag_dt["best_reduction"] * 100.0,
        "final_generation": res_ag_dt["final_generation"],
        "best_generation_found": res_ag_dt["best_generation_found"],
        "total_time_seconds": res_ag_dt["total_time_seconds"],
        "evaluations_requested": res_ag_dt["cache_stats"]["evaluations_requested"],
        "evaluations_computed": res_ag_dt["cache_stats"]["evaluations_computed"],
        "cache_hits": res_ag_dt["cache_stats"]["cache_hits"],
        "cache_hit_rate_pct": res_ag_dt["cache_stats"]["cache_hit_rate_pct"]
    }])
    ag_dt_summary.to_csv(os.path.join(results_dir, "ag_dt_results.csv"), index=False)
    
    # 5. SEGUNDA EJECUCIÓN DEL ALGORITMO GENÉTICO: AG-RF (Subconjunto B)
    # Evaluador de fitness específico para Random Forest con su propia caché INDEPENDIENTE
    evaluator_rf = FitnessEvaluator(
        model_factory=lambda: get_random_forest(random_state=RANDOM_STATE),
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_val=preprocessed.X_val,
        y_val=preprocessed.y_val,
        n_total=n_total,
        model_name="Random Forest"
    )
    
    ag_rf = GeneticAlgorithm(
        n_total=n_total,
        pop_size=50,
        max_generations=50,
        stagnation_limit=10,
        tournament_size=3,
        crossover_prob=0.8,
        random_state=RANDOM_STATE
    )
    
    res_ag_rf = ag_rf.run(evaluator=evaluator_rf, experiment_name="AG-RF")
    
    # Guardar historial y subconjunto B
    res_ag_rf["history_df"].to_csv(os.path.join(results_dir, "ag_rf_history.csv"), index=False)
    save_selected_features(
        feature_names=active_features,
        chromosome=res_ag_rf["best_chromosome"],
        output_path=os.path.join(results_dir, "selected_features_ag_rf.csv"),
        ag_name="AG-RF"
    )
    
    # Guardar métricas resumen de AG-RF
    ag_rf_summary = pd.DataFrame([{
        "experiment": "AG-RF",
        "evaluator_model": "Random Forest",
        "best_fitness": res_ag_rf["best_fitness"],
        "best_recall_validation": res_ag_rf["best_recall"],
        "num_features_selected": res_ag_rf["best_num_features"],
        "total_features": n_total,
        "reduction_pct": res_ag_rf["best_reduction"] * 100.0,
        "final_generation": res_ag_rf["final_generation"],
        "best_generation_found": res_ag_rf["best_generation_found"],
        "total_time_seconds": res_ag_rf["total_time_seconds"],
        "evaluations_requested": res_ag_rf["cache_stats"]["evaluations_requested"],
        "evaluations_computed": res_ag_rf["cache_stats"]["evaluations_computed"],
        "cache_hits": res_ag_rf["cache_stats"]["cache_hits"],
        "cache_hit_rate_pct": res_ag_rf["cache_stats"]["cache_hit_rate_pct"]
    }])
    ag_rf_summary.to_csv(os.path.join(results_dir, "ag_rf_results.csv"), index=False)
    
    # 6. EVALUACIÓN FINAL CRUZADA SOBRE TEST (Módulo evaluation)
    # Evalúa Todas las features, Subconjunto A y Subconjunto B con DT, RF y XGBoost
    df_final = run_final_evaluation_matrix(
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_test=preprocessed.X_test,
        y_test=preprocessed.y_test,
        n_total=n_total,
        best_chrom_dt=res_ag_dt["best_chromosome"],
        best_chrom_rf=res_ag_rf["best_chromosome"],
        fitness_dt=res_ag_dt["best_fitness"],
        fitness_rf=res_ag_rf["best_fitness"],
        random_state=RANDOM_STATE
    )
    df_final.to_csv(os.path.join(results_dir, "final_results.csv"), index=False)
    print(f"\n[FINAL] Resultados completos guardados en: {os.path.join(results_dir, 'final_results.csv')}")
    
    # 7. GENERAR GRÁFICO DE CONVERGENCIA
    plot_convergence(
        history_dt=res_ag_dt["history_df"],
        history_rf=res_ag_rf["history_df"],
        output_path=os.path.join(results_dir, "convergence_comparison.png")
    )
    
    # 8. PRESENTACIÓN DE RESULTADOS CONSOLIDADOS (CONSOLA)
    print("\n" + "=" * 95)
    print("TABLA COMPARATIVA FINAL: RENDIMIENTO SOBRE EL CONJUNTO TEST")
    print("=" * 95)
    
    pivot_recall = df_final.pivot(index="feature_set", columns="model", values="recall")
    pivot_f1 = df_final.pivot(index="feature_set", columns="model", values="f1")
    pivot_acc = df_final.pivot(index="feature_set", columns="model", values="accuracy")
    pivot_train_t = df_final.pivot(index="feature_set", columns="model", values="training_time_sec")
    
    print("\n--- RECALL RANSOMWARE (GR=1) SOBRE TEST ---")
    print(pivot_recall.to_string())
    
    print("\n--- F1-SCORE RANSOMWARE (GR=1) SOBRE TEST ---")
    print(pivot_f1.to_string())
    
    print("\n--- ACCURACY GLOBAL SOBRE TEST ---")
    print(pivot_acc.to_string())
    
    print("\n--- TIEMPO DE ENTRENAMIENTO (SEGUNDOS) ---")
    print(pivot_train_t.to_string())
    
    print("\n--- RESUMEN DE REDUCCIÓN DE CARACTERÍSTICAS Y TIEMPOS DE AG ---")
    reduction_summary = pd.DataFrame([
        {
            "Experimento": "Todas las características (Baseline)",
            "Características": n_total,
            "Reducción (%)": "0.0%",
            "Tiempo AG (s)": "N/A",
            "Evaluaciones Totales": "N/A",
            "Ahorro Caché (%)": "N/A"
        },
        {
            "Experimento": "Subconjunto AG-DT (Optimizado con DT)",
            "Características": res_ag_dt["best_num_features"],
            "Reducción (%)": f"{res_ag_dt['best_reduction']*100:.2f}%",
            "Tiempo AG (s)": f"{res_ag_dt['total_time_seconds']:.2f}s ({res_ag_dt['total_time_seconds']/60:.2f} min)",
            "Evaluaciones Totales": res_ag_dt["cache_stats"]["evaluations_requested"],
            "Ahorro Caché (%)": f"{res_ag_dt['cache_stats']['cache_hit_rate_pct']:.1f}%"
        },
        {
            "Experimento": "Subconjunto AG-RF (Optimizado con RF)",
            "Características": res_ag_rf["best_num_features"],
            "Reducción (%)": f"{res_ag_rf['best_reduction']*100:.2f}%",
            "Tiempo AG (s)": f"{res_ag_rf['total_time_seconds']:.2f}s ({res_ag_rf['total_time_seconds']/60:.2f} min)",
            "Evaluaciones Totales": res_ag_rf["cache_stats"]["evaluations_requested"],
            "Ahorro Caché (%)": f"{res_ag_rf['cache_stats']['cache_hit_rate_pct']:.1f}%"
        }
    ])
    print(reduction_summary.to_string(index=False))
    print("=" * 95)
    print("\n[EXPERIMENTO FINALIZADO CON ÉXITO]")


if __name__ == "__main__":
    main()
