import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split


class DatosPreprocesados:
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        dropped_constant_features: List[str],
        n_total: int
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_names
        self.dropped_constant_features = dropped_constant_features
        self.n_total = n_total


def preprocesar_datos(df: pd.DataFrame, random_state: int = 42) -> DatosPreprocesados:
    
    print("\n" + "=" * 50)
    print("INICIO DEL PREPROCESAMIENTO")
    print("=" * 50)
    y = df["objetivo"].astype(int)
    candidate_cols = [str(i) for i in range(1024)]
    X_candidate = df[candidate_cols].copy()
    excluded_cols = ["identificador", "nombre_archivo", "familia"]
    for col in excluded_cols:
        assert col not in X_candidate.columns, f"Error metodológico: '{col}' no debe estar en X."
    print(f"[PREPROCESAMIENTO] Columnas excluidas exitosamente de las características: {excluded_cols}")
    assert X_candidate.isnull().sum().sum() == 0, "Error: Se detectaron valores faltantes en las características."
    print(f"[PREPROCESAMIENTO] Cantidad inicial de características candidatas: {X_candidate.shape[1]}")
    nunique_per_col = X_candidate.nunique()
    constant_features = nunique_per_col[nunique_per_col == 1].index.tolist()
    
    print(f"[PREPROCESAMIENTO] Características constantes detectadas: {len(constant_features)}")
    if constant_features:
        print(f"[PREPROCESAMIENTO] Lista de características constantes eliminadas: {constant_features}")
        for col in constant_features:
            val = X_candidate[col].iloc[0]
            print(f"   -> Característica '{col}' eliminada (valor constante = {val})")
    X_cleaned = X_candidate.drop(columns=constant_features)
    n_total = X_cleaned.shape[1]
    active_feature_names = X_cleaned.columns.tolist()
    print(f"[PREPROCESAMIENTO] Características restantes tras limpieza (N_total): {n_total}")
    print(f"[PREPROCESAMIENTO] REGLA CRÍTICA: Longitud del cromosoma del AG = {n_total} genes.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_cleaned,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )
    total_samples = len(df)
    print("\n[PREPROCESAMIENTO] Distribución de conjuntos (Estratificada por GR):")
    print(
        f"  - Entrenamiento: {len(X_train)} muestras ({len(X_train)/total_samples*100:.1f}%) | "
        f"Goodware (0)={sum(y_train==0)}, Ransomware (1)={sum(y_train==1)}"
    )
    print(
        f"  - Validación:    {len(X_val)} muestras ({len(X_val)/total_samples*100:.1f}%) | "
        f"Goodware (0)={sum(y_val==0)}, Ransomware (1)={sum(y_val==1)}"
    )
    print(
        f"  - Prueba:        {len(X_test)} muestras ({len(X_test)/total_samples*100:.1f}%) | "
        f"Goodware (0)={sum(y_test==0)}, Ransomware (1)={sum(y_test==1)}"
    )
    print("[PREPROCESAMIENTO] El conjunto de prueba queda aislado exclusivamente para la evaluación final.")
    print("=" * 50 + "\n")
    
    return DatosPreprocesados(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=active_feature_names,
        dropped_constant_features=constant_features,
        n_total=n_total
    )


if __name__ == "__main__":
    from src.cargador_datos import cargar_datos
    df_raw, _ = cargar_datos("data/encabezados_ransomware.csv")
    data = preprocesar_datos(df_raw, random_state=42)
    print(f"[PRUEBA DE PREPROCESAMIENTO] N_total = {data.n_total}")
