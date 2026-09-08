

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.cargador_datos import cargar_datos
from src.preprocesamiento import preprocesar_datos
from src.modelos import obtener_arbol_decision, obtener_bosque_aleatorio
from src.aptitud import EvaluadorAptitud
from src.algoritmo_genetico import AlgoritmoGenetico
from src.evaluacion import ejecutar_evaluacion_referencia, ejecutar_matriz_evaluacion_final


def guardar_caracteristicas_seleccionadas(
    feature_names: list,
    chromosome: np.ndarray,
    output_path: str,
    ag_name: str
) -> pd.DataFrame:
    
    selected = [
        {"indice_caracteristica_activa": idx, "nombre_caracteristica": feature_names[idx]}
        for idx, gene in enumerate(chromosome) if gene == 1
    ]
    df_sel = pd.DataFrame(selected)
    df_sel.to_csv(output_path, index=False)
    print(f"[{ag_name}] Subconjunto guardado en: {output_path} ({len(df_sel)} características)")
    return df_sel


def graficar_convergencia(history_dt: pd.DataFrame, history_rf: pd.DataFrame, output_path: str):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes[0].plot(history_dt["generacion"], history_dt["mejor_aptitud"], label="Mejor Fitness", color="#1f77b4", lw=2)
    axes[0].plot(history_dt["generacion"], history_dt["aptitud_promedio"], label="Fitness Promedio", color="#aec7e8", linestyle="--")
    axes[0].plot(history_dt["generacion"], history_dt["peor_aptitud"], label="Peor Fitness", color="#c7c7c7", linestyle=":")
    axes[0].set_title("Evolución AG-DT (Evaluador: Decision Tree)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Generación", fontsize=11)
    axes[0].set_ylabel("Aptitud (0.7*Recall + 0.3*Reducción)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[1].plot(history_rf["generacion"], history_rf["mejor_aptitud"], label="Mejor Fitness", color="#2ca02c", lw=2)
    axes[1].plot(history_rf["generacion"], history_rf["aptitud_promedio"], label="Fitness Promedio", color="#98df8a", linestyle="--")
    axes[1].plot(history_rf["generacion"], history_rf["peor_aptitud"], label="Peor Fitness", color="#c7c7c7", linestyle=":")
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
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    data_path = "data/encabezados_ransomware.csv"
    df_raw, data_info = cargar_datos(data_path)
    RANDOM_STATE = 42
    preprocessed = preprocesar_datos(df_raw, random_state=RANDOM_STATE)
    
    n_total = preprocessed.n_total
    active_features = preprocessed.feature_names
    df_baseline = ejecutar_evaluacion_referencia(
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_test=preprocessed.X_test,
        y_test=preprocessed.y_test,
        n_total=n_total,
        random_state=RANDOM_STATE
    )
    df_baseline.to_csv(os.path.join(results_dir, "resultados_referencia.csv"), index=False)
    print(f"[REFERENCIA] Resultados guardados en: {os.path.join(results_dir, 'resultados_referencia.csv')}")
    evaluator_dt = EvaluadorAptitud(
        model_factory=lambda: obtener_arbol_decision(random_state=RANDOM_STATE),
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_val=preprocessed.X_val,
        y_val=preprocessed.y_val,
        n_total=n_total,
        model_name="Arbol de decision"
    )
    
    ag_dt = AlgoritmoGenetico(
        total_caracteristicas=n_total,
        tamano_poblacion=50,
        max_generaciones=50,
        limite_estancamiento=10,
        tamano_torneo=3,
        probabilidad_cruce=0.8,
        semilla_aleatoria=RANDOM_STATE
    )
    
    res_ag_dt = ag_dt.ejecutar(evaluador=evaluator_dt, nombre_experimento="AG-DT")
    res_ag_dt["historial"].to_csv(os.path.join(results_dir, "historial_ag_dt.csv"), index=False)
    guardar_caracteristicas_seleccionadas(
        feature_names=active_features,
        chromosome=res_ag_dt["mejor_cromosoma"],
        output_path=os.path.join(results_dir, "caracteristicas_seleccionadas_ag_dt.csv"),
        ag_name="AG-DT"
    )
    ag_dt_summary = pd.DataFrame([{
        "experimento": "AG-DT",
        "modelo_evaluador": "Arbol de decision",
        "mejor_aptitud": res_ag_dt["mejor_aptitud"],
        "mejor_recall_validacion": res_ag_dt["mejor_recall"],
        "numero_caracteristicas_seleccionadas": res_ag_dt["mejor_num_caracteristicas"],
        "total_caracteristicas": n_total,
        "reduccion_porcentaje": res_ag_dt["mejor_reduccion"] * 100.0,
        "generacion_final": res_ag_dt["generacion_final"],
        "mejor_generacion_encontrada": res_ag_dt["mejor_generacion_encontrada"],
        "tiempo_total_segundos": res_ag_dt["tiempo_total_segundos"],
        "evaluaciones_solicitadas": res_ag_dt["estadisticas_cache"]["evaluations_requested"],
        "evaluaciones_calculadas": res_ag_dt["estadisticas_cache"]["evaluations_computed"],
        "aciertos_cache": res_ag_dt["estadisticas_cache"]["cache_hits"],
        "porcentaje_aciertos_cache": res_ag_dt["estadisticas_cache"]["cache_hit_rate_pct"]
    }])
    ag_dt_summary.to_csv(os.path.join(results_dir, "resultados_ag_dt.csv"), index=False)
    evaluator_rf = EvaluadorAptitud(
        model_factory=lambda: obtener_bosque_aleatorio(random_state=RANDOM_STATE),
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_val=preprocessed.X_val,
        y_val=preprocessed.y_val,
        n_total=n_total,
        model_name="Random Forest"
    )
    
    ag_rf = AlgoritmoGenetico(
        total_caracteristicas=n_total,
        tamano_poblacion=50,
        max_generaciones=50,
        limite_estancamiento=10,
        tamano_torneo=3,
        probabilidad_cruce=0.8,
        semilla_aleatoria=RANDOM_STATE
    )
    
    res_ag_rf = ag_rf.ejecutar(evaluador=evaluator_rf, nombre_experimento="AG-RF")
    res_ag_rf["historial"].to_csv(os.path.join(results_dir, "historial_ag_rf.csv"), index=False)
    guardar_caracteristicas_seleccionadas(
        feature_names=active_features,
        chromosome=res_ag_rf["mejor_cromosoma"],
        output_path=os.path.join(results_dir, "caracteristicas_seleccionadas_ag_rf.csv"),
        ag_name="AG-RF"
    )
    ag_rf_summary = pd.DataFrame([{
        "experimento": "AG-RF",
        "modelo_evaluador": "Random Forest",
        "mejor_aptitud": res_ag_rf["mejor_aptitud"],
        "mejor_recall_validacion": res_ag_rf["mejor_recall"],
        "numero_caracteristicas_seleccionadas": res_ag_rf["mejor_num_caracteristicas"],
        "total_caracteristicas": n_total,
        "reduccion_porcentaje": res_ag_rf["mejor_reduccion"] * 100.0,
        "generacion_final": res_ag_rf["generacion_final"],
        "mejor_generacion_encontrada": res_ag_rf["mejor_generacion_encontrada"],
        "tiempo_total_segundos": res_ag_rf["tiempo_total_segundos"],
        "evaluaciones_solicitadas": res_ag_rf["estadisticas_cache"]["evaluations_requested"],
        "evaluaciones_calculadas": res_ag_rf["estadisticas_cache"]["evaluations_computed"],
        "aciertos_cache": res_ag_rf["estadisticas_cache"]["cache_hits"],
        "porcentaje_aciertos_cache": res_ag_rf["estadisticas_cache"]["cache_hit_rate_pct"]
    }])
    ag_rf_summary.to_csv(os.path.join(results_dir, "resultados_ag_rf.csv"), index=False)
    df_final = ejecutar_matriz_evaluacion_final(
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_test=preprocessed.X_test,
        y_test=preprocessed.y_test,
        n_total=n_total,
        best_chrom_dt=res_ag_dt["mejor_cromosoma"],
        best_chrom_rf=res_ag_rf["mejor_cromosoma"],
        fitness_dt=res_ag_dt["mejor_aptitud"],
        fitness_rf=res_ag_rf["mejor_aptitud"],
        random_state=RANDOM_STATE
    )
    df_final.to_csv(os.path.join(results_dir, "resultados_finales.csv"), index=False)
    print(f"\n[FINAL] Resultados completos guardados en: {os.path.join(results_dir, 'resultados_finales.csv')}")
    graficar_convergencia(
        history_dt=res_ag_dt["historial"],
        history_rf=res_ag_rf["historial"],
        output_path=os.path.join(results_dir, "convergence_comparison.png")
    )
    print("\n" + "=" * 95)
    print("TABLA COMPARATIVA FINAL: RENDIMIENTO SOBRE EL CONJUNTO DE PRUEBA")
    print("=" * 95)
    
    pivot_recall = df_final.pivot(index="conjunto_caracteristicas", columns="modelo", values="recall")
    pivot_f1 = df_final.pivot(index="conjunto_caracteristicas", columns="modelo", values="puntuacion_f1")
    pivot_acc = df_final.pivot(index="conjunto_caracteristicas", columns="modelo", values="exactitud")
    pivot_train_t = df_final.pivot(index="conjunto_caracteristicas", columns="modelo", values="tiempo_entrenamiento_seg")
    
    print("\n--- RECALL DE RANSOMWARE (GR=1) SOBRE EL CONJUNTO DE PRUEBA ---")
    print(pivot_recall.to_string())
    
    print("\n--- PUNTUACIÓN F1 DE RANSOMWARE (GR=1) SOBRE EL CONJUNTO DE PRUEBA ---")
    print(pivot_f1.to_string())
    
    print("\n--- EXACTITUD GLOBAL SOBRE EL CONJUNTO DE PRUEBA ---")
    print(pivot_acc.to_string())
    
    print("\n--- TIEMPO DE ENTRENAMIENTO (SEGUNDOS) ---")
    print(pivot_train_t.to_string())
    
    print("\n--- RESUMEN DE REDUCCIÓN DE CARACTERÍSTICAS Y TIEMPOS DE AG ---")
    reduction_summary = pd.DataFrame([
        {
            "Experimento": "Todas las características (Referencia)",
            "Características": n_total,
            "Reducción (%)": "0.0%",
            "Tiempo AG (s)": "N/A",
            "Evaluaciones Totales": "N/A",
            "Ahorro Caché (%)": "N/A"
        },
        {
            "Experimento": "Subconjunto AG-DT (Optimizado con DT)",
            "Características": res_ag_dt["mejor_num_caracteristicas"],
            "Reducción (%)": f"{res_ag_dt['mejor_reduccion']*100:.2f}%",
            "Tiempo AG (s)": f"{res_ag_dt['tiempo_total_segundos']:.2f}s ({res_ag_dt['tiempo_total_segundos']/60:.2f} min)",
            "Evaluaciones Totales": res_ag_dt["estadisticas_cache"]["evaluations_requested"],
            "Ahorro Caché (%)": f"{res_ag_dt['estadisticas_cache']['cache_hit_rate_pct']:.1f}%"
        },
        {
            "Experimento": "Subconjunto AG-RF (Optimizado con RF)",
            "Características": res_ag_rf["mejor_num_caracteristicas"],
            "Reducción (%)": f"{res_ag_rf['mejor_reduccion']*100:.2f}%",
            "Tiempo AG (s)": f"{res_ag_rf['tiempo_total_segundos']:.2f}s ({res_ag_rf['tiempo_total_segundos']/60:.2f} min)",
            "Evaluaciones Totales": res_ag_rf["estadisticas_cache"]["evaluations_requested"],
            "Ahorro Caché (%)": f"{res_ag_rf['estadisticas_cache']['cache_hit_rate_pct']:.1f}%"
        }
    ])
    print(reduction_summary.to_string(index=False))
    print("=" * 95)
    print("\n[EXPERIMENTO FINALIZADO CON ÉXITO]")


if __name__ == "__main__":
    main()
