# Cambios y decisiones tomadas en la implementación

Este documento resume **únicamente las decisiones y cambios realizados después de definir el marco teórico**, es decir, las decisiones que terminaron de darle forma a la implementación experimental.

## 1. Cambio de dataset

Se reemplazó el dataset que se había considerado inicialmente por:

**Ransomware PE Header Feature Dataset**

El cambio se realizó porque este dataset está directamente alineado con el enfoque definido en el marco teórico: **detección de ransomware mediante características estáticas del formato PE**.

El dataset terminó teniendo **1024 características originales**, de las cuales **6 fueron eliminadas por ser constantes**, quedando:

> **1018 características efectivamente utilizables por el AG.**

Un punto importante que se decidió durante el diseño es que **N nunca debe estar hardcodeado**. El número de genes del cromosoma se obtiene dinámicamente después del preprocesamiento:

```text
N = cantidad de features después de la limpieza
```

Por lo tanto, si en el futuro se eliminan más características constantes, el algoritmo se adapta automáticamente.

---

## 2. Modelos utilizados

Inicialmente se habían considerado:

* Random Forest
* Decision Tree
* LSTM

Durante el diseño se descartó **LSTM**, ya que no resulta apropiado para nuestro problema basado en características estáticas de archivos PE.

Finalmente se incorporó:

* **Decision Tree**
* **Random Forest**
* **XGBoost**

La idea es utilizar **Decision Tree y Random Forest como evaluadores del AG**, mientras que los tres modelos se utilizan posteriormente para evaluar la capacidad de transferencia de los subconjuntos seleccionados.

Esto permite estudiar no solamente si el AG encuentra un buen subconjunto, sino también si ese subconjunto continúa siendo útil cuando se utiliza con otro modelo de Machine Learning.

---

## 3. Se definieron dos ejecuciones del Algoritmo Genético

En lugar de utilizar un único AG, se decidió realizar **dos optimizaciones independientes**:

```text
AG-DT → fitness evaluado mediante Decision Tree
AG-RF → fitness evaluado mediante Random Forest
```

Esto es asi porque el fitness necesita evaluar si el cromosoma es sirve para detectar ransomware o no.

Posteriormente, ambos subconjuntos se evalúan con:

```text
Decision Tree
Random Forest
XGBoost
```

Por lo tanto:

```text
                    Dataset
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           AG-DT               AG-RF
             │                   │
          430 features        440 features
             │                   │
             └─────────┬─────────┘
                       ▼
              Evaluación final
                DT / RF / XGB
```

---

## 4. Fitness

Se estableció definitivamente la función:

$$
Fitness = 0.7 \times Recall + 0.3 \times Reduction
$$

donde:

* **Recall** representa el porcentaje de los ransomware reales fueron identificados como ransomware.
* **Reduction** representa la proporción de características eliminadas.

La prioridad es deliberada:

```text
70 % → capacidad de detectar ransomware
30 % → reducción de características
```

Esto refleja el objetivo del proyecto: **reducir características sin sacrificar significativamente la detección de ransomware**.

---

## 5. Validación y Test

Se decidió separar claramente la optimización de la evaluación final:

```text
Train
   +
Validation
   ↓
Optimización del AG
```

y:

```text
Test
↓
Evaluación final
```

El **Test nunca participa en el cálculo del fitness ni en la selección del individuo**.

Esto permite comparar posteriormente los subconjuntos seleccionados contra el baseline utilizando datos que permanecieron aislados durante la optimización.

También se decidió **no utilizar Cross-Validation dentro del AG inicialmente**, debido al costo computacional adicional.

---

## 6. Hiperparámetros de los modelos

Para garantizar reproducibilidad, los hiperparámetros quedaron fijados:

### Decision Tree

```python
criterion="gini"
random_state=42
```

### Random Forest

```python
n_estimators=100
criterion="gini"
n_jobs=-1
random_state=42
```

### XGBoost

```python
n_estimators=100
max_depth=6
learning_rate=0.1
random_state=42
```

No se realizará optimización de hiperparámetros durante el experimento.

---

## 7. Parámetros del AG

Se establecieron los siguientes parámetros:

| Parámetro                 | Decisión                       |
| ------------------------- | ------------------------------ |
| Población                 | **50 individuos**              |
| Máximo de generaciones    | **50**                         |
| Selección                 | **Torneo, k = 3**              |
| Elitismo                  | **1 individuo**                |
| Crossover                 | **Monopunto**                  |
| Probabilidad de crossover | **0,8**                        |
| Mutación                  | **1 / (2,5 × N)**              |
| Criterio de estancamiento | **10 generaciones sin mejora** |
| Cromosoma                 | Binario                        |


**N es dinámico**, por lo que en nuestro dataset actual:

```text
N = 1018
Pmutacion ≈ 0,000393
```

---

## 8. Caché de Fitness (a revisar)

Se incorporó una **caché de individuos evaluados**.

Si el AG genera nuevamente exactamente el mismo cromosoma:

```text
cromosoma → fitness ya calculado
```

no se vuelve a entrenar el modelo.

En su lugar:

```text
Cromosoma repetido
       ↓
¿Está en caché?
   ┌───┴───┐
   │       │
  Sí      No
   │       │
fitness   entrenar ML
guardado     ↓
           fitness
             ↓
           guardar
```

Esto evita entrenamientos innecesarios, especialmente durante las últimas generaciones, cuando la población comienza a converger y aparecen individuos repetidos.

---

## 9. Criterio de parada

Se mantuvieron dos condiciones:

1. **Máximo de 50 generaciones.**
2. **Parada anticipada después de 10 generaciones consecutivas sin mejorar el mejor fitness.**

Esto permite que el AG continúe buscando mejoras mientras las encuentre, pero evita gastar recursos cuando la población se ha estancado.

En la primera ejecución:

* **AG-DT** llegó hasta las 50 generaciones.
* **AG-RF** se detuvo en la generación 39 debido al estancamiento durante 10 generaciones.

---

## 10. Evaluación final

Después de obtener los dos mejores individuos:

```text
AG-DT → subconjunto A
AG-RF → subconjunto B
```

ambos subconjuntos se evalúan nuevamente sobre el **Test aislado** utilizando los tres modelos:

```text
             ┌── Decision Tree
AG-DT ───────┼── Random Forest
             └── XGBoost

             ┌── Decision Tree
AG-RF ───────┼── Random Forest
             └── XGBoost
```

Además, se mantiene un **Baseline**, entrenando los mismos modelos utilizando las 1018 características.

La comparación permite determinar:

* cuánto se redujeron las características;
* cuánto cambió el Recall de ransomware;
* cuánto cambió F1-score y Accuracy;
* cuánto cambió el tiempo de entrenamiento;
* cómo se comporta cada subconjunto al transferirse a otros modelos.

---

# 11. Qué buscamos demostrar con esta implementación

> **Encontrar subconjuntos significativamente más pequeños que mantengan la capacidad de detección de ransomware y reduzcan el costo computacional, y analizar además si el modelo utilizado para guiar el AG influye en la utilidad del subconjunto obtenido.**

Por eso tenemos:

```text
                REDUCCIÓN
                    +
             RECALL RANSOMWARE
                    +
          COSTO COMPUTACIONAL
                    +
       TRANSFERENCIA ENTRE MODELOS
```

Esta última parte es especialmente importante porque nos permite comparar **AG-DT frente a AG-RF**, en lugar de obtener simplemente un único subconjunto y declarar que "el AG funcionó".
