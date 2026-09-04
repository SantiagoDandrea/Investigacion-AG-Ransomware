"""
Módulo preprocessing.py
Responsabilidad: Limpiar dataset, excluir metadatos, detectar y eliminar características constantes,
y realizar división estratificada en Train (80%), Validation (10%) y Test (10%).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split


class PreprocessedData:
    """
    Estructura que encapsula los subconjuntos procesados y la información de características.
    """
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


def preprocess_data(df: pd.DataFrame, random_state: int = 42) -> PreprocessedData:
    """
    Ejecuta el preprocesamiento del dataset siguiendo estrictamente las reglas metodológicas:
    1. Identifica 'GR' como target (0 = Goodware, 1 = Ransomware).
    2. Selecciona las columnas '0' a '1023' como características candidatas.
    3. Excluye 'ID', 'filename' y 'family' (no se usan como features).
    4. Comprueba y elimina características completamente constantes.
    5. Define dinámicamente N_total como la cantidad de características sobrevivientes.
    6. Divide el dataset de forma estratificada: 80% Train, 10% Validation, 10% Test.
    
    Parámetros:
        df (pd.DataFrame): DataFrame original.
        random_state (int): Semilla para garantizar reproducibilidad (default: 42).
        
    Retorna:
        PreprocessedData: Objeto con los datos divididos e información de features.
    """
    print("\n" + "=" * 50)
    print("INICIO DEL PREPROCESAMIENTO")
    print("=" * 50)
    
    # 1. Separar target
    y = df["GR"].astype(int)
    
    # 2. Seleccionar columnas de features candidatas ('0' a '1023')
    candidate_cols = [str(i) for i in range(1024)]
    X_candidate = df[candidate_cols].copy()
    
    # 3. Comprobar exclusión estricta de ID, filename y family
    excluded_cols = ["ID", "filename", "family"]
    for col in excluded_cols:
        assert col not in X_candidate.columns, f"Error metodológico: '{col}' no debe estar en X."
    print(f"[PREPROCESAMIENTO] Columnas excluidas exitosamente de features: {excluded_cols}")
    
    # 4. Comprobar tipos de datos y valores faltantes
    assert X_candidate.isnull().sum().sum() == 0, "Error: Se detectaron valores faltantes en features."
    print(f"[PREPROCESAMIENTO] Cantidad inicial de características candidatas: {X_candidate.shape[1]}")
    
    # 5. Comprobar y eliminar características completamente constantes
    # Una columna constante tiene número de valores únicos igual a 1 (o std == 0)
    nunique_per_col = X_candidate.nunique()
    constant_features = nunique_per_col[nunique_per_col == 1].index.tolist()
    
    print(f"[PREPROCESAMIENTO] Características constantes detectadas: {len(constant_features)}")
    if constant_features:
        print(f"[PREPROCESAMIENTO] Lista de características constantes eliminadas: {constant_features}")
        for col in constant_features:
            val = X_candidate[col].iloc[0]
            print(f"   -> Feature '{col}' eliminada (valor constante = {val})")
            
    # Eliminar constantes
    X_cleaned = X_candidate.drop(columns=constant_features)
    
    # 6. Definir N_total dinámicamente
    n_total = X_cleaned.shape[1]
    active_feature_names = X_cleaned.columns.tolist()
    print(f"[PREPROCESAMIENTO] Características restantes tras limpieza (N_total): {n_total}")
    print(f"[PREPROCESAMIENTO] REGLA CRÍTICA: Longitud del cromosoma del AG = {n_total} genes.")
    
    # 7. División estratificada según GR: 80% Train, 10% Validation, 10% Test
    # Paso 1: Separar 80% Train y 20% Temporal (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_cleaned,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    
    # Paso 2: Dividir el 20% temporal en 50% Validation y 50% Test (cada uno es 10% del total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Verificación de proporciones y estratificación
    total_samples = len(df)
    print("\n[PREPROCESAMIENTO] Distribución de conjuntos (Estratificada por GR):")
    print(
        f"  - Train:      {len(X_train)} muestras ({len(X_train)/total_samples*100:.1f}%) | "
        f"Goodware (0)={sum(y_train==0)}, Ransomware (1)={sum(y_train==1)}"
    )
    print(
        f"  - Validation: {len(X_val)} muestras ({len(X_val)/total_samples*100:.1f}%) | "
        f"Goodware (0)={sum(y_val==0)}, Ransomware (1)={sum(y_val==1)}"
    )
    print(
        f"  - Test:       {len(X_test)} muestras ({len(X_test)/total_samples*100:.1f}%) | "
        f"Goodware (0)={sum(y_test==0)}, Ransomware (1)={sum(y_test==1)}"
    )
    print("[PREPROCESAMIENTO] El conjunto Test queda aislado exclusivamente para la evaluación final.")
    print("=" * 50 + "\n")
    
    return PreprocessedData(
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
    from src.data_loader import load_raw_data
    df_raw, _ = load_raw_data("data/Ransomware_headers.csv")
    data = preprocess_data(df_raw, random_state=42)
    print(f"[TEST PREPROCESSING] N_total = {data.n_total}")
