from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from typing import Tuple

def load_data() -> pd.DataFrame:
    """
    Carga el conjunto de datos de Cáncer de Mama de Wisconsin desde sklearn y lo convierte en un DataFrame.
    
    Return:
        pd.DataFrame: El conjunto de datos conteniendo características y el objetivo.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def get_feature_target_split(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Divide el DataFrame en características (X) y objetivo (y).
    
    Args:
        df: DataFrame de entrada.
        target_col: Nombre de la columna objetivo.
        
    Return:
        Tuple[pd.DataFrame, pd.Series]: X e y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
