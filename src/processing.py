from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Tuple

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba con estratificación.
    
    ¿Por qué estratificar?
    La estratificación asegura que la proporción de clases en los conjuntos de entrenamiento y prueba 
    coincida con el conjunto de datos original. Esto es crucial para conjuntos de datos desequilibrados o tamaños de muestra pequeños
    para evitar sesgos en la evaluación.
    
    Args:
        X: Características.
        y: Objetivo.
        test_size: Proporción del conjunto de datos para incluir en la división de prueba.
        random_state: Semilla para reproducibilidad.
        
    Return:
        Tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def create_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Crea un pipeline de scikit-learn con escalado y el modelo dado.
    
    ¿Por qué usar Pipeline?
    Los pipelines encapsulan los pasos de preprocesamiento (como el escalado) y el modelo en un solo objeto.
    Esto evita la fuga de datos (por ejemplo, calcular la media/desviación estándar en el conjunto de prueba) y simplifica 
    el despliegue y la validación cruzada.
    
    Args:
        model: El estimador de scikit-learn para incluir en el pipeline.
        
    Return:
        Pipeline: El pipeline construido.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
