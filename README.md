# Optimización de la Selección de Características para la Detección de Ransomware mediante Algoritmos Genéticos

Proyecto de investigación universitario enfocado en evaluar si un **Algoritmo Genético (AG)** puede seleccionar automáticamente un subconjunto reducido de características estáticas extraídas de los primeros 1024 bytes del **PE Header (Portable Executable)** de Windows, manteniendo una capacidad de detección de ransomware comparable a la obtenida utilizando todas las características, y evaluando el impacto sobre el costo computacional.

## 1. Estructura del conjunto de datos

- **Archivo**: `data/encabezados_ransomware.csv`
- **Total de muestras**: 2157
  - **Goodware (`GR = 0`)**: 1134 muestras (52.57%)
  - **Ransomware (`GR = 1`)**: 1023 muestras (47.43%)
  - **Familias de ransomware**: 25 familias (codificadas en la columna `family`)
- **Columnas de metadatos (excluidas estrictamente del entrenamiento y de las características)**:
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

Para evitar fugas de información, el conjunto de datos se divide de forma estratificada conservando la proporción de clases:

- **Entrenamiento (80%)**: 1725 muestras (907 Goodware, 818 Ransomware). Utilizado para entrenar los clasificadores durante la optimización.
- **Validación (10%)**: 216 muestras (113 Goodware, 103 Ransomware). Utilizado exclusivamente para evaluar la función de aptitud de los individuos en el AG.
- **Prueba (10%)**: 216 muestras (114 Goodware, 102 Ransomware). **Aislado completamente** durante toda la búsqueda del AG. Solo se utiliza en la evaluación final.
- Semilla fija: `random_state = 42`.

---

## 3. Modelos de aprendizaje automático e hiperparámetros fijos

No se realiza optimización de hiperparámetros , manteniéndose fijos en todas las etapas:
| Modelo | Hiperparámetros | Rol en el Proyecto |
| :--- | :--- | :--- |
| **Decision Tree (DT)** | `criterion="gini"`, `random_state=42` | Evaluador de fitness en **AG-DT** y modelo de evaluación final. |
| **Random Forest (RF)** | `n_estimators=100`, `criterion="gini"`, `n_jobs=-1`, `random_state=42` | Evaluador de fitness en **AG-RF** y modelo de evaluación final. |
| **XGBoost (XGB)** | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, `eval_metric="logloss"` | Modelo de evaluación final para comprobar transferibilidad de características. |

---

## 4. Configuración del Algoritmo Genético

El AG busca un subconjunto óptimo de características maximizando el recall de ransomware y minimizando la cantidad de características.

- **Representación**: Cromosoma binario de longitud $N_{total} = 1018$.
  - `1`: Característica seleccionada.
  - `0`: Característica descartada.

### Parámetros utilizados

- Tamaño de la población: `50` individuos.
- Número máximo de generaciones: `50`.
- Límite de estancamiento: `10` generaciones sin mejora.
- Tamaño del torneo: `3` individuos.
- Probabilidad de cruce: `0.8`.
- Probabilidad de mutación: `1 / (2.5 * N_total)`.
- Semilla aleatoria: `42`, para facilitar la reproducibilidad.

En cada generación, el algoritmo evalúa los cromosomas, conserva el mejor individuo y crea una nueva población mediante selección por torneo, cruce y mutación. La función de aptitud combina el recall de ransomware obtenido en validación con la reducción de características:

$$
aptitud = 0.7 \cdot recall + 0.3 \cdot reduccion
$$

## 5. ¿Qué hace el código?

El flujo principal se encuentra en `src/main.py` y sigue estas etapas:

1. Carga el archivo `data/encabezados_ransomware.csv` y comprueba que existan las columnas de metadatos y las 1024 características esperadas.
2. Separa la variable objetivo (`objetivo`) de las características y excluye los metadatos (`identificador`, `nombre_archivo` y `familia`).
3. Elimina las características constantes y divide los datos de forma estratificada en entrenamiento, validación y prueba.
4. Calcula una evaluación de referencia utilizando todas las características.
5. Ejecuta un algoritmo genético con un árbol de decisión (AG-DT).
6. Ejecuta otro algoritmo genético con un bosque aleatorio (AG-RF).
7. Evalúa en el conjunto de prueba los subconjuntos encontrados por ambos algoritmos y los compara con el conjunto completo de características.
8. Guarda los resultados y genera una gráfica de convergencia.

Durante la optimización, el conjunto de prueba permanece aislado. Solo se utiliza al final para medir el rendimiento de los modelos y evitar que influya en la selección de características.

## 6. Estructura del proyecto

```text
Investigacion-AG-Ransomware/
├── data/
│   └── encabezados_ransomware.csv
├── results/
│   └── Resultados generados por el programa
├── src/
│   ├── algoritmo_genetico.py       # Selección, cruce, mutación y evolución
│   ├── aptitud.py                  # Evaluación de aptitud y caché
│   ├── cargador_datos.py           # Lectura y validación del CSV
│   ├── evaluacion.py               # Evaluación de referencia y evaluación final
│   ├── modelos.py                  # Árbol, bosque aleatorio y XGBoost
│   ├── preprocesamiento.py         # Limpieza y división de los datos
│   └── main.py                     # Punto de entrada del proyecto
├── requirements.txt
└── README.md
```

### Descripción de los módulos principales

- `cargador_datos.py`: carga el dataset y valida su estructura.
- `preprocesamiento.py`: prepara las características, elimina columnas constantes y realiza la división estratificada.
- `modelos.py`: define los clasificadores utilizados en la investigación.
- `aptitud.py`: entrena el modelo correspondiente para cada cromosoma, calcula el recall, la reducción y la aptitud. También utiliza una caché para evitar repetir evaluaciones.
- `algoritmo_genetico.py`: implementa la población, selección por torneo, cruce, mutación, elitismo y criterios de parada.
- `evaluacion.py`: calcula la evaluación de referencia y la comparación final sobre datos de prueba.
- `main.py`: coordina todo el experimento y guarda los archivos de salida.

## 7. Instalación

Se recomienda utilizar Python 3.10 o una versión posterior y un entorno virtual. Desde PowerShell, situado en la carpeta raíz del proyecto, ejecuta:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si el entorno virtual ya existe, basta con activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 8. Ejecución

Con el entorno virtual activado, ejecuta:

```powershell
python -m src.main
```

También puedes ejecutar el programa directamente usando el intérprete del entorno virtual:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

El programa debe ejecutarse desde la carpeta raíz del proyecto para que encuentre correctamente `data/encabezados_ransomware.csv` y la carpeta `results/`. La ejecución puede tardar varios minutos, ya que se entrenan numerosos modelos durante las generaciones de ambos algoritmos genéticos.

## 9. Archivos generados

Al finalizar, la carpeta `results/` contiene, entre otros, los siguientes archivos:

- `resultados_referencia.csv`: rendimiento utilizando todas las características.
- `historial_ag_dt.csv` y `historial_ag_rf.csv`: evolución de la aptitud por generación.
- `caracteristicas_seleccionadas_ag_dt.csv` y `caracteristicas_seleccionadas_ag_rf.csv`: características elegidas por cada AG.
- `resultados_ag_dt.csv` y `resultados_ag_rf.csv`: resumen de cada experimento genético.
- `resultados_finales.csv`: comparación final de los modelos y subconjuntos evaluados.
- `convergence_comparison.png`: gráfica de convergencia de AG-DT y AG-RF.

## 10. Interpretación general

El objetivo no es únicamente obtener el mayor recall posible, sino encontrar un equilibrio entre detectar correctamente el ransomware y utilizar menos características. Una reducción alta con un recall similar al de la referencia indica que el algoritmo genético encontró un subconjunto compacto con un rendimiento comparable. La evaluación final sobre el conjunto de prueba permite comprobar si ese resultado se mantiene con datos que no participaron durante la optimización.
