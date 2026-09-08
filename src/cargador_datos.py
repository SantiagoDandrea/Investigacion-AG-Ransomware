import os
import pandas as pd
from typing import Tuple, Dict, Any


def cargar_datos(file_path: str = "data/encabezados_ransomware.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo del dataset en: {file_path}")
    
    print(f"[CARGADOR DE DATOS] Cargando conjunto de datos desde: {file_path}...")
    df = pd.read_csv(file_path, sep=";", skiprows=[0])
    
    num_rows, num_cols = df.shape
    print(f"[CARGADOR DE DATOS] Conjunto de datos cargado con éxito: {num_rows} muestras y {num_cols} columnas.")
    required_meta_cols = ["identificador", "nombre_archivo", "objetivo", "familia"]
    missing_meta = [col for col in required_meta_cols if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Faltan columnas de metadatos requeridas: {missing_meta}")
    original_feature_cols = [str(i) for i in range(1024)]
    missing_features = [col for col in original_feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Faltan {len(missing_features)} columnas de características esperadas (0 a 1023)."
        )
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        print(f"[CARGADOR DE DATOS] ADVERTENCIA: Se detectaron {total_missing} valores nulos en el conjunto de datos.")
    else:
        print("[CARGADOR DE DATOS] Verificación de nulos completada: no se detectaron valores faltantes.")
    gr_counts = df["objetivo"].value_counts().to_dict()
    goodware_count = gr_counts.get(0, 0)
    ransomware_count = gr_counts.get(1, 0)
    print(f"[CARGADOR DE DATOS] Distribución de clases GR: Goodware (0) = {goodware_count}, Ransomware (1) = {ransomware_count}")

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
    df_raw, info_dict = cargar_datos("data/encabezados_ransomware.csv")
    print("[CARGADOR DE DATOS] Prueba finalizada exitosamente.")
