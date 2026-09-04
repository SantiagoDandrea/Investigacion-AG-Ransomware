"""
Módulo models.py
Responsabilidad: Crear instancias de Decision Tree, Random Forest y XGBoost
con los hiperparámetros EXACTOS especificados en el diseño experimental.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Union

ClassifierType = Union[DecisionTreeClassifier, RandomForestClassifier, XGBClassifier]


def get_decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """
    Retorna un DecisionTreeClassifier configurado según el diseño experimental:
    - criterion="gini"
    - random_state=42
    """
    return DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state
    )


def get_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """
    Retorna un RandomForestClassifier configurado según el diseño experimental:
    - n_estimators=100
    - criterion="gini"
    - n_jobs=-1
    - random_state=42
    """
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        n_jobs=-1,
        random_state=random_state
    )


def get_xgboost(random_state: int = 42) -> XGBClassifier:
    """
    Retorna un XGBClassifier configurado según el diseño experimental:
    - n_estimators=100
    - max_depth=6
    - learning_rate=0.1
    - random_state=42
    """
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss"
    )


def get_model_by_name(name: str, random_state: int = 42) -> ClassifierType:
    """
    Instancia el clasificador solicitado por su nombre clave.
    Valores admitidos: 'DT', 'RF', 'XGB'.
    """
    name_upper = name.strip().upper()
    if name_upper in ["DT", "DECISION TREE", "DECISIONTREE"]:
        return get_decision_tree(random_state=random_state)
    elif name_upper in ["RF", "RANDOM FOREST", "RANDOMFOREST"]:
        return get_random_forest(random_state=random_state)
    elif name_upper in ["XGB", "XGBOOST"]:
        return get_xgboost(random_state=random_state)
    else:
        raise ValueError(f"Modelo desconocido: '{name}'. Opciones válidas: 'DT', 'RF', 'XGB'.")


if __name__ == "__main__":
    dt = get_decision_tree()
    rf = get_random_forest()
    xgb = get_xgboost()
    print("[MODELS] Instanciación exitosa de los 3 clasificadores con parámetros fijos:")
    print(" - DT:", dt)
    print(" - RF:", rf)
    print(" - XGB:", xgb)
