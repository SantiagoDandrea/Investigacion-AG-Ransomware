# Optimización de la Selección de Características para la Detección de Ransomware mediante Algoritmos Genéticos

## Proyecto de investigación universitario enfocado en evaluar si un **Algoritmo Genético (AG)** puede seleccionar automáticamente un subconjunto reducido de características estáticas extraídas de los primeros 1024 bytes del **PE Header (Portable Executable)** de Windows, manteniendo una capacidad de detección de ransomware comparable a la obtenida utilizando todas las características, y evaluando el impacto sobre el costo computacional.

## 1. Estructura del Dataset

- **Archivo**: `data/Ransomware_headers.csv`
- **Total de muestras**: 2157
  - **Goodware (`GR = 0`)**: 1134 muestras (52.57%)
  - **Ransomware (`GR = 1`)**: 1023 muestras (47.43%)
  - **Familias de ransomware**: 25 familias (codificadas en la columna `family`)
- **Columnas de metadatos (Excluidas estrictamente del entrenamiento y de las features)**:
  - `ID`: Identificador de la muestra.
  - `filename`: Nombre del archivo ejecutable original.
  - `GR`: Variable objetivo binaria (`0 = Goodware`, `1 = Ransomware`).
  - `family`: Identificador de familia de ransomware.
- **Características originales**:
  - 1024 columnas numéricas (`'0'` a `'1023'`), correspondientes a los primeros 1024 bytes del encabezado PE (valores entre 0 y 255).
- **Limpieza de características constantes (Regla Crítica)**:
  - Se analizan todas las características y se eliminan aquellas con varianza cero (completamente constantes).
  - En este dataset se eliminaron 6 características constantes: `'0'` (77='M'), `'1'` (90='Z'), `'62'` (0), `'63'` (0), `'423'` (0) y `'447'` (0).
  - **$N_{total}$ dinámico**: **1018 características** ($1024 - 6$).
  - La longitud del cromosoma se define dinámicamente como **1018 genes**.

---

## 2. División de Datos (Estratificada)

Para evitar fugas de información (_data leakage_), el dataset se divide de forma estratificada conservando la proporción de clases:

- **Train (80%)**: 1725 muestras (907 Goodware, 818 Ransomware). Utilizado para entrenar los clasificadores durante la optimización.
- **Validation (10%)**: 216 muestras (113 Goodware, 103 Ransomware). Utilizado exclusivamente para evaluar la función de fitness de los individuos en el AG.
- **Test (10%)**: 216 muestras (114 Goodware, 102 Ransomware). **Aislado completamente** durante toda la búsqueda del AG. Solo se utiliza en la evaluación final.
- Semilla fija: `random_state = 42`.

---

## 3. Modelos de Machine Learning e Hiperparámetros Fijos

No se realiza optimización de hiperparámetros (Grid Search, Bayesian Search, etc.), manteniéndose fijos en todas las etapas:
| Modelo | Hiperparámetros | Rol en el Proyecto |
| :--- | :--- | :--- |
| **Decision Tree (DT)** | `criterion="gini"`, `random_state=42` | Evaluador de fitness en **AG-DT** y modelo de evaluación final. |
| **Random Forest (RF)** | `n_estimators=100`, `criterion="gini"`, `n_jobs=-1`, `random_state=42` | Evaluador de fitness en **AG-RF** y modelo de evaluación final. |
| **XGBoost (XGB)** | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, `eval_metric="logloss"` | Modelo de evaluación final para comprobar transferibilidad de características. |

---

## 4. Configuración del Algoritmo Genético

El AG busca un subconjunto óptimo de características maximizando el Recall de ransomware y minimizando la cantidad de características.

- **Representación**: Cromosoma binario de longitud $N_{total} = 1018$.
  - `1`: Característica seleccionada.
  - `0`: Característica descartada.
