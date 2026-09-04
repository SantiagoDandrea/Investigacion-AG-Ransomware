# Optimización de la Selección de Características para la Detección de Ransomware mediante Algoritmos Genéticos

Proyecto de investigación universitario enfocado en evaluar si un **Algoritmo Genético (AG)** puede seleccionar automáticamente un subconjunto reducido de características estáticas extraídas de los primeros 1024 bytes del **PE Header (Portable Executable)** de Windows, manteniendo una capacidad de detección de ransomware comparable a la obtenida utilizando todas las características, y evaluando el impacto sobre el costo computacional.

---

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

Para evitar fugas de información (*data leakage*), el dataset se divide de forma estratificada conservando la proporción de clases:
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
- **Población inicial**: 50 individuos generados aleatoriamente. Se garantiza que cada individuo posea al menos 1 característica seleccionada ($N_{selected} \ge 1$).
- **Función de Fitness**:
  $$\text{Fitness} = 0.7 \times \text{Recall}_{\text{GR}=1} + 0.3 \times \text{Reduction}$$
  $$\text{Reduction} = 1 - \frac{N_{\text{selected}}}{N_{total}}$$
  - El Recall se calcula **estrictamente sobre la clase positiva** $GR = 1$ (Ransomware) en el conjunto Validation.
  - El tiempo de cómputo **no** forma parte del fitness.
- **Caché de Evaluaciones**:
  - Implementada para evitar reentrenar modelos ante cromosomas idénticos.
  - Clave hashable: `tuple(chromosome)`.
  - Instancias independientes para AG-DT y AG-RF.
- **Operadores Genéticos**:
  - **Selección**: Torneo de tamaño $k = 3$.
  - **Cruce**: Single-point crossover con probabilidad $P_c = 0.8$.
  - **Mutación**: Bit-flip por gen independiente con probabilidad dinámica:
    $$P_m = \frac{1}{2.5 \times N_{total}} = \frac{1}{2.5 \times 1018} = \frac{1}{2545} \approx 0.0003929$$
    Se garantiza que tras la mutación el individuo tenga al menos 1 gen activo.
  - **Elitismo**: Exactamente 1 mejor individuo pasa intacto a la siguiente generación.
  - **Reemplazo**: Generacional (1 élite + 49 descendientes).
- **Criterios de Parada**:
  - Máximo de **50 generaciones**.
  - Parada temprana por estancamiento si el mejor fitness no mejora durante **10 generaciones consecutivas**.

---

## 5. Arquitectura del Código

El proyecto adopta una estructura modular y limpia:

```
Codigo-AG-PEHeaders/
├── data/
│   └── Ransomware_headers.csv       # Dataset principal
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Carga y validación estructural del dataset
│   ├── preprocessing.py             # Limpieza de constantes y split estratificado
│   ├── models.py                    # Factorías de modelos con hiperparámetros fijos
│   ├── fitness.py                   # Función de fitness y caché de evaluaciones
│   ├── genetic_algorithm.py         # Implementación del AG, operadores y parada
│   ├── evaluation.py                # Baseline y evaluación final sobre Test
│   └── main.py                      # Orquestador del flujo experimental completo
├── results/                         # Resultados exportados en CSV y gráficos
│   ├── baseline_results.csv         # Desempeño con todas las características
│   ├── ag_dt_results.csv            # Resumen de AG-DT
│   ├── ag_rf_results.csv            # Resumen de AG-RF
│   ├── ag_dt_history.csv            # Evolución generación a generación de AG-DT
│   ├── ag_rf_history.csv            # Evolución generación a generación de AG-RF
│   ├── selected_features_ag_dt.csv  # Índices y nombres de features (Subconjunto A)
│   ├── selected_features_ag_rf.csv  # Índices y nombres de features (Subconjunto B)
│   ├── final_results.csv            # Matriz comparativa final sobre Test
│   └── convergence_comparison.png   # Gráfica de convergencia del fitness
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Documentación académica
```

---

## 6. Procedimiento de Ejecución

### Requisitos del Sistema
- **Python 3.10+** (probado en Python 3.13)
- Entorno virtual recomendado.

### Instalación de Dependencias
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar el Experimento Completo
```bash
python -m src.main
```
El script ejecutará en secuencia el preprocesamiento, el baseline, las dos optimizaciones (AG-DT y AG-RF), la evaluación cruzada final sobre el conjunto Test, y generará automáticamente todos los reportes y gráficos en la carpeta `results/`.
