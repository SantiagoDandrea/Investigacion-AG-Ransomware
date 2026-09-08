

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Union

ClassifierType = Union[DecisionTreeClassifier, RandomForestClassifier, XGBClassifier]


def obtener_arbol_decision(random_state: int = 42) -> DecisionTreeClassifier:
    
    return DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state
    )


def obtener_bosque_aleatorio(random_state: int = 42) -> RandomForestClassifier:
    
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        n_jobs=-1,
        random_state=random_state
    )


def obtener_xgboost(random_state: int = 42) -> XGBClassifier:
    
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss"
    )


def obtener_modelo_por_nombre(name: str, random_state: int = 42) -> ClassifierType:
    
    name_upper = name.strip().upper()
    if name_upper in ["DT", "DECISION TREE", "DECISIONTREE"]:
        return obtener_arbol_decision(random_state=random_state)
    elif name_upper in ["RF", "RANDOM FOREST", "RANDOMFOREST"]:
        return obtener_bosque_aleatorio(random_state=random_state)
    elif name_upper in ["XGB", "XGBOOST"]:
        return obtener_xgboost(random_state=random_state)
    else:
        raise ValueError(f"Modelo desconocido: '{name}'. Opciones válidas: 'DT', 'RF', 'XGB'.")


if __name__ == "__main__":
    dt = obtener_arbol_decision()
    rf = obtener_bosque_aleatorio()
    xgb = obtener_xgboost()
    print("[MODELOS] Instanciación exitosa de los 3 clasificadores con parámetros fijos:")
    print(" - DT:", dt)
    print(" - RF:", rf)
    print(" - XGB:", xgb)
