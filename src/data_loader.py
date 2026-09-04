"""
Módulo data_loader.py
Responsabilidad: Cargar el dataset 'Ransomware_headers.csv' y comprobar su estructura básica.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_data(file_path: str = "data/Ransomware_headers.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga el archivo CSV del dataset de PE Headers y verifica su integridad estructural.
    
    Parámetros:
        file_path (str): Ruta al archivo CSV.
        
    Retorna:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - df: DataFrame con los datos originales.
            - info: Diccionario con estadísticas preliminares de inspección.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo del dataset en: {file_path}")
    
    print(f"[DATA LOADER] Cargando dataset desde: {file_path}...")
    df = pd.read_csv(file_path)
    
    num_rows, num_cols = df.shape
    print(f"[DATA LOADER] Dataset cargado con éxito: {num_rows} muestras y {num_cols} columnas.")
    
    # 1. Comprobar columnas de metadatos esperadas
    required_meta_cols = ["ID", "filename", "GR", "family"]
    missing_meta = [col for col in required_meta_cols if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Faltan columnas de metadatos requeridas: {missing_meta}")
    
    # 2. Comprobar presencia de características candidatas '0' a '1023'
    original_feature_cols = [str(i) for i in range(1024)]
    missing_features = [col for col in original_feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Faltan {len(missing_features)} columnas de características esperadas (0 a 1023)."
        )
        
    # 3. Comprobar valores faltantes (NaN/null)
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        print(f"[DATA LOADER] ADVERTENCIA: Se detectaron {total_missing} valores nulos en el dataset.")
    else:
        print("[DATA LOADER] Verificación de nulos completada: 0 valores faltantes detectados.")
        
    # 4. Distribución de la variable objetivo GR
    gr_counts = df["GR"].value_counts().to_dict()
    goodware_count = gr_counts.get(0, 0)
    ransomware_count = gr_counts.get(1, 0)
    print(f"[DATA LOADER] Distribución de clases GR: Goodware (0) = {goodware_count}, Ransomware (1) = {ransomware_count}")

    info = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "goodware_count": goodware_count,
        "ransomware_count": ransomware_count,
        "total_missing": total_missing,
        "meta_cols": required_meta_cols,
        "original_feature_cols": original_feature_cols
    }
    
    return df, info


if __name__ == "__main__":
    # Prueba individual del módulo
    df_raw, info_dict = load_raw_data("data/Ransomware_headers.csv")
    print("[DATA LOADER] Prueba finalizada exitosamente.")
